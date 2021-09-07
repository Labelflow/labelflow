import { useMemo, useRef } from "react";
import { useRouter } from "next/router";
import { Draw as OlDraw } from "ol/interaction";
import { createBox, DrawEvent } from "ol/interaction/Draw";
import GeoJSON from "ol/format/GeoJSON";
import { Fill, Stroke, Style } from "ol/style";
import GeometryType from "ol/geom/GeometryType";
import { useApolloClient, useQuery, gql } from "@apollo/client";

import { useToast } from "@chakra-ui/react";

import { useHotkeys } from "react-hotkeys-hook";
import { LabelType } from "@labelflow/graphql-types";
import {
  useLabellingStore,
  Tools,
  DrawingToolState,
} from "../../../../connectors/labeling-state";
import { keymap } from "../../../../keymap";
import { useUndoStore } from "../../../../connectors/undo-store";
import { noneClassColor } from "../../../../utils/class-color-generator";
import { createLabelEffect } from "./create-label-effect";

const labelClassQuery = gql`
  query getLabelClass($id: ID!) {
    labelClass(where: { id: $id }) {
      id
      name
      color
    }
  }
`;

const geometryFunction = createBox();

export const DrawBoundingBoxAndPolygonInteraction = () => {
  const drawRef = useRef<OlDraw>(null);
  const client = useApolloClient();
  const { imageId } = useRouter()?.query;

  const selectedTool = useLabellingStore((state) => state.selectedTool);

  const setDrawingToolState = useLabellingStore(
    (state) => state.setDrawingToolState
  );
  const setSelectedLabelId = useLabellingStore(
    (state) => state.setSelectedLabelId
  );
  const selectedLabelClassId = useLabellingStore(
    (state) => state.selectedLabelClassId
  );
  const { data: dataLabelClass } = useQuery(labelClassQuery, {
    variables: { id: selectedLabelClassId },
    skip: selectedLabelClassId == null,
  });
  const { perform } = useUndoStore();

  const selectedLabelClass = dataLabelClass?.labelClass;

  useHotkeys(
    keymap.cancelAction.key,
    () => drawRef.current?.abortDrawing(),
    {},
    [drawRef]
  );

  const toast = useToast();

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

  if (![Tools.BOX, Tools.POLYGON].includes(selectedTool)) {
    return null;
  }
  if (typeof imageId !== "string") {
    return null;
  }
  const interactionDrawArguments =
    selectedTool === Tools.POLYGON
      ? {
          type: GeometryType.MULTI_POLYGON,
          style, // Needed here to trigger the rerender of the component when the selected class changes
        }
      : {
          type: GeometryType.CIRCLE,
          geometryFunction,
          style, // Needed here to trigger the rerender of the component when the selected class changes
        };

  const errorMessage =
    selectedTool === Tools.POLYGON
      ? "Error creating polygon"
      : "Error creating bounding box";
  return (
    <olInteractionDraw
      ref={drawRef}
      args={interactionDrawArguments}
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
      onDrawend={async (drawEvent: DrawEvent) => {
        const geometry = new GeoJSON().writeGeometryObject(
          drawEvent.feature.getGeometry()
        ) as GeoJSON.Polygon;
        const createLabelPromise = perform(
          createLabelEffect(
            {
              imageId,
              selectedLabelClassId,
              geometry,
              labelType:
                selectedTool === Tools.POLYGON
                  ? LabelType.Polygon
                  : LabelType.Box,
            },
            {
              setSelectedLabelId,
              client,
            }
          )
        );
        setDrawingToolState(DrawingToolState.IDLE);
        try {
          await createLabelPromise;
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
      }}
    />
  );
};
