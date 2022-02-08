import { createCreateLabelClassEffect } from "./create-label-class";
import { useUndoStore } from "..";
import { useLabelingStore } from "../../labeling-state";
import { client } from "../../apollo-client/schema-client";

import { setupTestsWithLocalDatabase } from "../../../utils/setup-local-db-tests";

setupTestsWithLocalDatabase();

jest.mock("../../apollo-client/schema-client", () => {
  const original = jest.requireActual("../../apollo-client/schema-client");
  return {
    client: {
      ...original.client,
      clearStore: original.client.clearStore, // This needs to be passed like this otherwise the resulting object does not have the clearStore method
      mutate: jest.fn(() => {
        return { data: { createLabelClass: { id: "label class id" } } };
      }),
      query: jest.fn(() => {
        return {
          data: {
            labelClasses: [
              {
                id: "existing label class id",
                name: "existing label class",
                color: "0xaa45f7",
              },
            ],
            label: {
              id: "my label id",
              labelClass: {
                id: "previous label class id",
              },
            },
          },
        };
      }),
    },
  };
});

const { perform } = useUndoStore.getState();

describe("CreateLabelClassEffect", () => {
  beforeEach(async () => {
    jest.clearAllMocks();
    await perform(
      createCreateLabelClassEffect(
        {
          name: "new label class",
          color: "0xaa45f8",
          datasetId: "a dataset id",
          selectedLabelClassIdPrevious: "previous label class id",
        },
        { client }
      )
    );
  });

  it("creates the label class and update the labeling store", async () => {
    expect(client.mutate).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        variables: {
          data: expect.objectContaining({
            name: "new label class",
            color: "0xaa45f8",
            datasetId: "a dataset id",
          }),
        },
      })
    );

    // The id is generated by the effect. We retrieve it from the first mutation
    // which is "createLabelClass"
    const { id } = (client.mutate as jest.Mock).mock.calls[0][0].variables.data;

    expect(useLabelingStore.getState()).toMatchObject({
      selectedLabelClassId: id,
    });
  });

  it("undo the label class creation the update of the labeling store", async () => {
    await useUndoStore.getState().undo();

    // The id is generated by the effect. We retrieve it from the first mutation
    // which is "createLabelClass"
    const { id } = (client.mutate as jest.Mock).mock.calls[0][0].variables.data;

    expect(client.mutate).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        variables: {
          where: { id },
        },
      })
    );
    expect(useLabelingStore.getState()).toMatchObject({
      selectedLabelClassId: "previous label class id",
    });
  });

  it("redo the update of the label class of a label and the update of the labeling store", async () => {
    await useUndoStore.getState().undo();
    await useUndoStore.getState().redo();

    // The id is generated by the effect. We retrieve it from the first mutation
    // which is "createLabelClass"
    const { id } = (client.mutate as jest.Mock).mock.calls[0][0].variables.data;

    expect(client.mutate).toHaveBeenNthCalledWith(
      3,
      expect.objectContaining({
        variables: {
          data: {
            name: "new label class",
            color: "0xaa45f8",
            datasetId: "a dataset id",
            id,
          },
        },
      })
    );
    expect(useLabelingStore.getState()).toMatchObject({
      selectedLabelClassId: id,
    });
  });
});