import { useRef, useCallback } from "react";
import { Spinner, Center, ThemeProvider, Box } from "@chakra-ui/react";
import { useRouter } from "next/router";
import { RouterContext } from "next/dist/shared/lib/router-context";
import { Extent, getCenter } from "ol/extent";
import { Map as OlMap, View as OlView, MapBrowserEvent } from "ol";
import { Geometry } from "ol/geom";
import { Vector as OlSourceVector } from "ol/source";
import { Size } from "ol/size";
import * as Sentry from "@sentry/nextjs";
import memoize from "mem";
import Projection from "ol/proj/Projection";
import useMeasure from "react-use-measure";
import { ApolloProvider, useApolloClient, useQuery, gql } from "@apollo/client";

import { Map } from "@labelflow/react-openlayers-fiber";
import type { Image } from "@labelflow/graphql-types";
import "ol/ol.css";

import { DrawInteraction } from "./draw-interaction";
import { SelectAndModifyFeature } from "./select-and-modify-feature";
import { ClassificationContent, ClassificationOverlay } from "./classification";
import { Labels } from "./labels";
import { EditLabelClass } from "./edit-label-class";
import { CursorGuides } from "./cursor-guides";
import {
  useLabelingStore,
  Tools,
  DrawingToolState,
} from "../../../connectors/labeling-state";
import { theme } from "../../../theme";
import { useImagePrefecthing } from "../../../hooks/use-image-prefetching";

const empty: any[] = [];

/*
 * Padding around the openlayers view
 * [top, right, bottom, left] in pixels
 * See https://openlayers.org/en/latest/apidoc/module-ol_View-View.html#padding
 */
const viewPadding = [72, 72, 72, 72];

/*
 * Standard projection, the same for all images, with arbitrary extent
 */
const standardProjection = new Projection({
  code: "standardImageStaticProjection",
  units: "pixels",
});

/*
 * Memoize openlayers parameters that we pass to the open layers components
 */
const getMemoizedProperties = memoize(
  (
    _imageId,
    image: Pick<Image, "id" | "url" | "width" | "height"> | null | undefined
  ) => {
    if (image == null) return {};
    const { url, width, height, id } = image;
    const size: Size = [width, height];
    const extent: Extent = [0, 0, width, height];
    const center = getCenter(extent);
    const projection = standardProjection;
    // We could also use an image-specific projection, as in openlayers examples:
    // It seems that we don't need it for now, but we might find that having a single global projection
    /// creates problem later on. So for now let's keep this option commented
    //     const projection = new Projection({
    //       code: imageId,
    //       units: "pixels",
    //       extent,
    //     });

    return { id, url, width, height, size, extent, center, projection };
  }
);

const imageQuery = gql`
  query image($id: ID!) {
    image(where: { id: $id }) {
      id
      width
      height
      url
    }
  }
`;

export const OpenlayersMap = () => {
  const editClassOverlayRef = useRef<HTMLDivElement | null>(null);
  const classificationOverlayRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<OlMap>(null);
  const viewRef = useRef<OlView | null>(null);
  const sourceVectorBoxesRef = useRef<OlSourceVector<Geometry> | null>(null);
  const router = useRouter();
  const { imageId } = router?.query;
  const isContextMenuOpen = useLabelingStore(
    (state) => state.isContextMenuOpen
  );
  const setIsContextMenuOpen = useLabelingStore(
    (state) => state.setIsContextMenuOpen
  );
  const selectedTool = useLabelingStore((state) => state.selectedTool);
  const setIsImageLoading = useLabelingStore(
    (state) => state.setIsImageLoading
  );
  const boxDrawingToolState = useLabelingStore(
    (state) => state.boxDrawingToolState
  );
  const setCanZoomIn = useLabelingStore((state) => state.setCanZoomIn);
  const setCanZoomOut = useLabelingStore((state) => state.setCanZoomOut);

  const setView = useLabelingStore((state) => state.setView);
  const zoomFactor = useLabelingStore((state) => state.zoomFactor);

  const { data: imageData, previousData: imageDataPrevious } = useQuery<{
    image: Pick<Image, "id" | "url" | "width" | "height">;
  }>(imageQuery, {
    variables: { id: imageId },
    skip: !imageId,
  });

  const image = imageData?.image ?? imageDataPrevious?.image;

  useImagePrefecthing();

  const client = useApolloClient();
  const [containerRef, bounds] = useMeasure();

  const isBoundsValid = bounds.width > 0 || bounds.height > 0;
  const onPointermove = useCallback(
    (e: MapBrowserEvent<UIEvent>) => {
      const mapTargetViewport = e.map.getViewport();
      if (!mapTargetViewport) return;
      if (e.dragging) {
        mapTargetViewport.style.cursor = "grabbing";
      } else if (selectedTool === Tools.BOX) {
        mapTargetViewport.style.cursor = "crosshair";
      } else if (selectedTool === Tools.POLYGON) {
        mapTargetViewport.style.cursor = "crosshair";
      } else if (selectedTool === Tools.SELECTION) {
        const hit = e.map.hasFeatureAtPixel(e.pixel);
        mapTargetViewport.style.cursor = hit ? "pointer" : "grab";
      } else {
        mapTargetViewport.style.cursor = "default";
      }
    },
    [selectedTool]
  );

  const memoizedImage = getMemoizedProperties(image?.id, image);

  const { url, size, extent, center, projection, width, height } =
    memoizedImage;

  const resolution =
    width && height
      ? Math.max(
          width / (bounds.width - viewPadding[1] - viewPadding[3]),
          height / (bounds.height - viewPadding[0] - viewPadding[2])
        )
      : 1;

  return (
    <Box
      sx={{
        display: "flex",
        width: "100%",
        height: "100%",
        "& .pointereventsnone": {
          // This !importsant is needed to take over the "pointer-events: auto" put by openlayers
          // overlays, for example on the classifications tags
          pointerEvents: "none !important",
        },
      }}
      onContextMenu={(e) => {
        e.preventDefault();
        return false;
      }}
    >
      <Map
        ref={mapRef}
        args={{ controls: empty }}
        style={{
          height: "100%",
          width: "100%",
          // Touch action none fixes a bug with shitty touch experience in openlayers
          // See https://github.com/openlayers/openlayers/issues/10757
          touchAction: "none",
        }}
        onPointermove={onPointermove}
        containerRef={containerRef}
      >
        {/* Need to bridge contexts across renderers
         * See https://github.com/facebook/react/issues/17275 */}
        <RouterContext.Provider value={router}>
          <ApolloProvider client={client}>
            <ThemeProvider theme={theme}>
              {
                // Before useMeasure has time to properly measure the div, we have a negative resolution,
                // There is no point rendering the view in that case
                isBoundsValid && (
                  <olView
                    ref={(value: OlView) => {
                      if (!value) return;
                      if (viewRef.current !== value) {
                        viewRef.current = value;
                        setView(value);
                      }
                    }}
                    onChange_resolution={() => {
                      if (!viewRef.current) return false;
                      setCanZoomIn(
                        viewRef.current.getZoom() + zoomFactor <
                          viewRef.current.getMaxZoom()
                      );
                      setCanZoomOut(
                        viewRef.current.getZoom() - zoomFactor >
                          viewRef.current.getMinZoom()
                      );
                      return false;
                    }}
                    args={{
                      extent,
                      maxResolution: resolution,
                      // Max zoom = 16 pixels of screen per pixel of image
                      minResolution: 1.0 / 16.0,
                    }}
                    center={center}
                    initialProjection={projection}
                    resolution={resolution}
                    constrainOnlyCenter
                    showFullExtent
                    padding={viewPadding}
                  />
                )
              }
              <olLayerImage extent={extent}>
                {url != null && (
                  <olSourceImageStatic
                    // ol/source/image does not have `setXXX` methods, only options in the constructor, so
                    // to change anything, you need to recreate the object. So we pass all in args.
                    // See https://openlayers.org/en/latest/apidoc/module-ol_source_Image.ImageSourceEvent.html
                    args={{
                      url,
                      imageExtent: extent,
                      imageSize: size,
                      projection,
                      crossOrigin: "anonymous",
                    }}
                    onImageloaderror={() => {
                      Sentry.captureException(
                        new Error(
                          "Image load error in openlayers. See https://github.com/labelflow/labelflow/issues/431"
                        ),
                        { extra: { url } }
                      );
                      // To solve a rare bug where image does not load
                      // See https://github.com/labelflow/labelflow/issues/431
                      console.warn(
                        "Reloading window to prevent bug https://github.com/labelflow/labelflow/issues/431"
                      );
                      window?.location?.reload?.();
                      return true;
                    }}
                    onImageloadstart={() => {
                      setIsImageLoading(true);
                      return true;
                    }}
                    onImageloadend={() => {
                      setIsImageLoading(false);
                      return true;
                    }}
                  />
                )}
              </olLayerImage>

              <Labels sourceVectorLabelsRef={sourceVectorBoxesRef} />
              <ClassificationOverlay
                image={memoizedImage}
                classificationOverlayRef={classificationOverlayRef}
              />
              <DrawInteraction />
              <SelectAndModifyFeature
                editClassOverlayRef={editClassOverlayRef}
                sourceVectorLabelsRef={sourceVectorBoxesRef}
                setIsContextMenuOpen={setIsContextMenuOpen}
                image={memoizedImage}
                map={mapRef.current}
              />

              {sourceVectorBoxesRef.current && (
                <olInteractionSnap source={sourceVectorBoxesRef.current} />
              )}
            </ThemeProvider>
          </ApolloProvider>
        </RouterContext.Provider>
      </Map>
      {[Tools.BOX, Tools.IOG].includes(selectedTool) &&
        boxDrawingToolState !== DrawingToolState.DRAWING &&
        !isContextMenuOpen && <CursorGuides map={mapRef.current} />}
      {/* This div is needed to prevent a weird error that seems related to the EditLabelClass component */}
      <Box
        key="toto"
        sx={{
          position: "absolute",
          pointerEvents: "none",
          height: "100%",
          width: "100%",
        }}
      >
        {url == null && (
          <Center h="full">
            <Spinner aria-label="loading indicator" size="xl" />
          </Center>
        )}
      </Box>

      <EditLabelClass
        key="EditLabelClass"
        ref={(e) => {
          if (e && editClassOverlayRef.current !== e) {
            editClassOverlayRef.current = e;
          }
        }}
        isOpen={isContextMenuOpen}
        onClose={() => setIsContextMenuOpen(false)}
      />
      <ClassificationContent
        key="ClassificationContent"
        ref={(e) => {
          if (e && classificationOverlayRef.current !== e) {
            classificationOverlayRef.current = e;
          }
        }}
      />
    </Box>
  );
};