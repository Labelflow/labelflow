import { useMemo, useRef, useCallback, useState, useEffect } from "react";
import { Draw as OlDraw } from "ol/interaction";
import { createBox, DrawEvent } from "ol/interaction/Draw";
import GeoJSON, { GeoJSONPolygon } from "ol/format/GeoJSON";
import { Fill, Stroke, Style } from "ol/style";
import { Vector as OlSourceVector } from "ol/source";
import Collection from "ol/Collection";
import GeometryType from "ol/geom/GeometryType";
import { Feature, MapBrowserEvent } from "ol";
import { Geometry, Polygon, Point } from "ol/geom";
import { useApolloClient, useQuery, gql } from "@apollo/client";
import { useToast } from "@chakra-ui/react";
import { useHotkeys } from "react-hotkeys-hook";
import CircleStyle from "ol/style/Circle";
import { LabelType } from "@labelflow/graphql-types";
import { ModifyEvent } from "ol/interaction/Modify";
import { useMap } from "@labelflow/react-openlayers-fiber";
import BaseEvent from "ol/events/Event";
import {
  useLabelingStore,
  DrawingToolState,
} from "../../../../connectors/labeling-state";
import { keymap } from "../../../../keymap";
import { noneClassColor } from "../../../../utils/class-color-generator";
import { createCreateLabelEffect } from "../../../../connectors/undo-store/effects/create-label";
import { createRunIogEffect } from "../../../../connectors/undo-store/effects/run-iog";
import { useUndoStore } from "../../../../connectors/undo-store";

const labelClassQuery = gql`
  query getLabelClass($id: ID!) {
    labelClass(where: { id: $id }) {
      id
      name
      color
    }
  }
`;

const imageQuery = gql`
  query getImage($id: ID!) {
    image(where: { id: $id }) {
      id
      url
    }
  }
`;

const geometryFunction = createBox();
type Coordinate = [number, number];

export const DrawIogInteraction = ({ imageId }: { imageId: string }) => {
  const map = useMap();
  const drawRef = useRef<OlDraw>(null);
  const client = useApolloClient();

  const { perform } = useUndoStore();
  const [pointsInside, setPointsInside] = useState<Coordinate[]>([]);
  const [pointsOutside, setPointsOutside] = useState<Coordinate[]>([]);
  const [centerPoint, setCenterPoint] = useState<Coordinate | null>(null);
  const vectorSourceRef = useRef<OlSourceVector<Geometry>>(null);
  const [centerPointFeature, setCenterPointFeature] =
    useState<Feature<Polygon> | null>(null);

  useEffect(() => {
    if (vectorSourceRef.current != null) {
      const centerPointFeatureFromSource =
        vectorSourceRef.current.getFeatureById("centerPoint");
      if (centerPoint != null)
        setCenterPointFeature(centerPointFeatureFromSource as Feature<Polygon>);

      const listener = (event: BaseEvent) => {
        const { feature } = event as unknown as { feature: Feature<Geometry> };
        if (feature.getProperties().id === "centerPoint") {
          setCenterPointFeature(feature as Feature<Polygon>);
        }
      };
      vectorSourceRef.current?.on(["addfeature"], listener);
      return () => vectorSourceRef.current?.un("addfeature", listener);
    }
    return () => {};
  }, [vectorSourceRef.current]);

  const setDrawingToolState = useLabelingStore(
    (state) => state.setDrawingToolState
  );
  const setSelectedLabelId = useLabelingStore(
    useCallback((state) => state.setSelectedLabelId, [])
  );
  const selectedLabelClassId = useLabelingStore(
    (state) => state.selectedLabelClassId
  );
  const { data: dataLabelClass } = useQuery(labelClassQuery, {
    variables: { id: selectedLabelClassId },
    skip: selectedLabelClassId == null,
  });
  const { data: dataImage } = useQuery(imageQuery, {
    variables: { id: imageId },
    skip: imageId == null,
  });

  const selectedLabelClass = dataLabelClass?.labelClass;

  const selectedLabelId = useLabelingStore((state) => state.selectedLabelId);

  const clearIogFeatures = useCallback(() => {
    setPointsInside([]);
    setPointsOutside([]);
    setCenterPoint(null);
    setSelectedLabelId(null);
  }, [setPointsInside, setPointsOutside, setCenterPoint, setSelectedLabelId]);
  useHotkeys(keymap.validateIogLabel.key, clearIogFeatures, {}, [
    clearIogFeatures,
  ]);
  useEffect(() => {
    clearIogFeatures();
  }, [imageId, clearIogFeatures]);

  useHotkeys(
    keymap.cancelAction.key,
    () => drawRef.current?.abortDrawing(),
    {},
    [drawRef]
  );

  const toast = useToast();

  const errorMessage = "Error executing IOG";
  useEffect(() => {
    if (pointsInside.length > 0 || pointsOutside.length > 0)
      perform(
        createRunIogEffect(
          {
            labelId: selectedLabelId ?? "",
            pointsInside,
            pointsOutside,
          },
          { client }
        )
      );
  }, [pointsInside, pointsOutside, selectedLabelId]);

  const style = useMemo(() => {
    const color = selectedLabelClass?.color ?? noneClassColor;

    return new Style({
      fill: new Fill({
        color: `${color}10`,
      }),
      stroke: new Stroke({
        color,
        width: 2,
      }),
    });
  }, [selectedLabelClass?.color]);

  const createPointInsideOrOutside = useCallback(
    (event: MapBrowserEvent<UIEvent>) => {
      const { map: mapEvent } = event;

      const idOfClickedFeature = mapEvent.forEachFeatureAtPixel(
        event.pixel,
        (feature) => feature.getProperties().id
      );

      if (idOfClickedFeature === selectedLabelId) {
        setPointsOutside((previousPoints) => [
          ...previousPoints,
          event.coordinate as Coordinate,
        ]);
      } else {
        setPointsInside((previousPoints) => [
          ...previousPoints,
          event.coordinate as Coordinate,
        ]);
      }

      return false;
    },
    [selectedLabelId]
  );

  useEffect(() => {
    if (selectedLabelId != null) {
      map?.on("click", createPointInsideOrOutside);
      return () => map?.un("click", createPointInsideOrOutside);
    }
    return () => {};
  }, [map, selectedLabelId, createPointInsideOrOutside]);

  const performIOGFromDrawEvent = async (drawEvent: DrawEvent) => {
    const openLayersGeometry = drawEvent.feature.getGeometry();
    const geometry = new GeoJSON().writeGeometryObject(
      openLayersGeometry
    ) as GeoJSONPolygon;
    const labelId = await createCreateLabelEffect(
      {
        imageId,
        selectedLabelClassId,
        geometry,
        labelType: LabelType.Polygon,
      },
      {
        setSelectedLabelId,
        client,
      }
    ).do();
    const inferencePromise = (async () => {
      const dataUrl = await (async () => {
        const blob = await fetch(dataImage?.image?.url).then((r) => r.blob());
        return new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.readAsDataURL(blob);
        });
      })();
      const [x, y, xMax, yMax] = openLayersGeometry.getExtent();
      const boundingBoxCenterPoint: Coordinate = [
        Math.floor((x + xMax) / 2),
        Math.floor((y + yMax) / 2),
      ];

      setCenterPoint(boundingBoxCenterPoint);

      return perform(
        createRunIogEffect(
          {
            labelId,
            x,
            y,
            height: yMax - y,
            width: xMax - x,
            imageUrl: dataUrl,
            centerPoint: boundingBoxCenterPoint,
          },
          { client }
        )
      );
    })();

    setDrawingToolState(DrawingToolState.IDLE);
    try {
      await inferencePromise;
    } catch (error) {
      toast({
        title: errorMessage,
        description: error?.message,
        isClosable: true,
        status: "error",
        position: "bottom-right",
        duration: 10000,
      });
    }
  };

  const performIogFromModifyEvent = async (
    modifyEvent: ModifyEvent
  ): Promise<boolean> => {
    const newCoordinates = modifyEvent.mapBrowserEvent.coordinate;
    setCenterPoint(newCoordinates);
    await perform(
      createRunIogEffect(
        {
          labelId: selectedLabelId ?? "",
          centerPoint: newCoordinates,
          pointsInside,
          pointsOutside,
        },
        { client }
      )
    );
    return true;
  };

  if (typeof imageId !== "string") {
    return null;
  }

  return selectedLabelId == null ? (
    <olInteractionDraw
      ref={drawRef}
      args={{
        type: GeometryType.CIRCLE,
        geometryFunction,
        style, // Needed here to trigger the rerender of the component when the selected class changes
      }}
      condition={(e) => {
        // 0 is the main mouse button. See: https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/button
        // @ts-ignore
        return e.originalEvent.button === 0;
      }}
      onDrawabort={() => {
        setDrawingToolState(DrawingToolState.IDLE);
        return true;
      }}
      onDrawstart={() => {
        setDrawingToolState(DrawingToolState.DRAWING);
        return true;
      }}
      onDrawend={performIOGFromDrawEvent}
    />
  ) : (
    <>
      {centerPointFeature != null && (
        <olInteractionModify
          args={{ features: new Collection([centerPointFeature]) }}
          onModifyend={performIogFromModifyEvent}
        />
      )}

      <olLayerVector>
        <olSourceVector ref={vectorSourceRef}>
          {[
            ...pointsInside.map((coordinates) => {
              return (
                <olFeature
                  key={coordinates.join("-")}
                  geometry={new Point(coordinates)}
                  style={
                    new Style({
                      image: new CircleStyle({
                        radius: 8,
                        fill: new Fill({
                          color: "#23e623ff",
                        }),
                        stroke: new Stroke({
                          color: "#ffffffff",
                          width: 2,
                        }),
                      }),
                    })
                  }
                />
              );
            }),
            ...pointsOutside.map((coordinates) => {
              return (
                <olFeature
                  key={coordinates.join("-")}
                  geometry={new Point(coordinates)}
                  style={
                    new Style({
                      image: new CircleStyle({
                        radius: 8,
                        fill: new Fill({
                          color: "#ff2323ff",
                        }),
                        stroke: new Stroke({
                          color: "#ffffffff",
                          width: 2,
                        }),
                      }),
                    })
                  }
                />
              );
            }),
            centerPoint ? (
              <olFeature
                key={centerPoint.join("-")}
                id="centerPoint"
                geometry={new Point(centerPoint)}
                style={
                  new Style({
                    image: new CircleStyle({
                      radius: 8,
                      fill: new Fill({
                        color: "#0023ffff",
                      }),
                      stroke: new Stroke({
                        color: "#ffffffff",
                        width: 2,
                      }),
                    }),
                  })
                }
              />
            ) : null,
          ].filter((item) => item)}
        </olSourceVector>
      </olLayerVector>
    </>
  );
};
