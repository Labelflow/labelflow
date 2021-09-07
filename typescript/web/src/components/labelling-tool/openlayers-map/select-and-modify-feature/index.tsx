import { MutableRefObject, useEffect, useState, useCallback } from "react";
import { Feature, Map as OlMap } from "ol";
import { Geometry, Polygon } from "ol/geom";
import { Vector as OlSourceVector } from "ol/source";
import Collection from "ol/Collection";
import { extend } from "@labelflow/react-openlayers-fiber";
import { ApolloClient, useApolloClient, gql, useQuery } from "@apollo/client";
import { useToast, UseToastOptions } from "@chakra-ui/react";
import { ModifyEvent } from "ol/interaction/Modify";
import { TranslateEvent } from "ol/interaction/Translate";
import { useRouter } from "next/router";
import { LabelType } from "@labelflow/graphql-types";
import { SelectInteraction } from "./select-interaction";
import {
  Tools,
  useLabellingStore,
} from "../../../../connectors/labeling-state";
import {
  ResizeAndTranslateBox,
  ResizeAndTranslateEvent,
} from "./resize-and-translate-box-interaction";
import { Effect, useUndoStore } from "../../../../connectors/undo-store";
import { updateLabelEffect } from "./update-label-effect";

// Extend react-openlayers-catalogue to include resize and translate interaction
extend({
  ResizeAndTranslateBox: { object: ResizeAndTranslateBox, kind: "Interaction" },
});

const getLabelQuery = gql`
  query getLabel($id: ID!) {
    label(where: { id: $id }) {
      type
      id
      geometry {
        type
        coordinates
      }
      labelClass {
        id
        color
      }
    }
  }
`;

export const interactionEnd = async (
  e: TranslateEvent | ModifyEvent | ResizeAndTranslateEvent | null,
  perform: (effect: Effect<any>) => Promise<void>,
  client: ApolloClient<Object>,
  imageId: string,
  toast: (options: UseToastOptions) => void
) => {
  const feature = e?.features?.item(0) as Feature<Polygon>;
  if (feature != null) {
    const coordinates = feature.getGeometry().getCoordinates();
    const geometry = { type: "Polygon", coordinates };
    const { id: labelId } = feature.getProperties();
    try {
      await perform(
        updateLabelEffect(
          {
            labelId,
            geometry,
            imageId,
          },
          { client }
        )
      );
    } catch (error) {
      toast({
        title: "Error updating label",
        description: error?.message,
        isClosable: true,
        status: "error",
        position: "bottom-right",
        duration: 10000,
      });
    }
  }
  return true;
};

export const SelectAndModifyFeature = (props: {
  sourceVectorLabelsRef: MutableRefObject<OlSourceVector<Geometry> | null>;
  map: OlMap | null;
  setIsContextMenuOpen?: (state: boolean) => void;
  editClassOverlayRef?: MutableRefObject<HTMLDivElement | null>;
}) => {
  const { sourceVectorLabelsRef } = props;
  const router = useRouter();
  const imageId = router?.query?.imageId as string;

  // We need to have this state in order to store the selected feature in the addfeature listener below
  const [selectedFeature, setSelectedFeature] =
    useState<Feature<Polygon> | null>(null);
  const selectedLabelId = useLabellingStore((state) => state.selectedLabelId);
  const selectedTool = useLabellingStore((state) => state.selectedTool);

  const { data: labelData } = useQuery(getLabelQuery, {
    variables: { id: selectedLabelId },
    skip: selectedLabelId == null,
  });

  const getSelectedFeature = useCallback(() => {
    if (selectedFeature?.getProperties()?.id !== selectedLabelId) {
      if (selectedLabelId == null) {
        setSelectedFeature(null);
      } else {
        const featureFromSource = sourceVectorLabelsRef.current
          ?.getFeatures()
          ?.filter(
            (feature) => feature.getProperties().id === selectedLabelId
          )?.[0];
        if (featureFromSource != null) {
          setSelectedFeature(featureFromSource as Feature<Polygon>);
        }
      }
    }
  }, [selectedLabelId, sourceVectorLabelsRef.current]);

  // This is needed to make sure that each time a new feature is added to OL we check if it's the selected feature (for instance when we reload the page and we have a selected label but labels haven't been added to OL yet)
  useEffect(() => {
    sourceVectorLabelsRef.current?.on("addfeature", getSelectedFeature);
    return () =>
      sourceVectorLabelsRef.current?.un("addfeature", getSelectedFeature);
  }, [sourceVectorLabelsRef.current, selectedLabelId]);

  useEffect(() => {
    getSelectedFeature();
  }, [selectedLabelId]);

  const client = useApolloClient();
  const { perform } = useUndoStore();
  const toast = useToast();
  return (
    <>
      <SelectInteraction {...props} />

      {selectedTool === Tools.SELECTION &&
        labelData?.label?.type === LabelType.Box && (
          /* @ts-ignore - We need to add this because resizeAndTranslateBox is not included in the react-openalyers-fiber original catalogue */
          <resizeAndTranslateBox
            args={{ selectedFeature }}
            onInteractionEnd={async (e: ResizeAndTranslateEvent | null) => {
              return await interactionEnd(e, perform, client, imageId, toast);
            }}
          />
        )}
      {selectedTool === Tools.SELECTION &&
        labelData?.label?.type === LabelType.Polygon &&
        selectedFeature && (
          <>
            <olInteractionTranslate
              args={{ features: new Collection([selectedFeature]) }}
              onTranslateend={async (e: TranslateEvent | null) => {
                return await interactionEnd(e, perform, client, imageId, toast);
              }}
            />
            <olInteractionModify
              args={{ features: new Collection([selectedFeature]) }}
              onModifyend={async (e: ModifyEvent | null) => {
                return await interactionEnd(e, perform, client, imageId, toast);
              }}
            />
            <olInteractionPointer
              args={{
                handleMoveEvent: (e) => {
                  const mapTargetViewport = e.map.getViewport();
                  if (mapTargetViewport != null) {
                    const clonedFeature = selectedFeature.clone();
                    clonedFeature.getGeometry().scale(0.95);
                    if (
                      clonedFeature
                        ?.getGeometry()
                        ?.intersectsCoordinate(e.coordinate) ??
                      false
                    ) {
                      mapTargetViewport.style.cursor = "move";
                    }
                  }
                },
              }}
            />
          </>
        )}
    </>
  );
};
