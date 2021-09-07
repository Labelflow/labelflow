import { gql, ApolloClient } from "@apollo/client";

import { LabelClass } from "@labelflow/graphql-types";
import { useLabellingStore } from "../../labeling-state";
import {
  getNextClassColor,
  hexColorSequence,
} from "../../../utils/class-color-generator";
import { Effect } from "..";
import { getDatasetsQuery } from "../../../pages/local/datasets";
import { datasetLabelClassesQuery } from "../../../components/dataset-class-list/class-item";

const createLabelClassQuery = gql`
  mutation createLabelClass($data: LabelClassCreateInput!) {
    createLabelClass(data: $data) {
      id
    }
  }
`;

const deleteLabelClassQuery = gql`
  mutation deleteLabelClass($where: LabelClassWhereUniqueInput!) {
    deleteLabelClass(where: $where) {
      id
    }
  }
`;

export const createCreateLabelClassEffect = (
  {
    name,
    color,
    datasetId,
    datasetSlug,
    selectedLabelClassIdPrevious,
  }: {
    name: string;
    color: string;
    datasetId: string;
    datasetSlug: string;
    selectedLabelClassIdPrevious: string | null;
  },
  {
    client,
  }: {
    client: ApolloClient<object>;
  }
): Effect => ({
  do: async () => {
    const {
      data: {
        createLabelClass: { id: labelClassId },
      },
    } = await client.mutate({
      mutation: createLabelClassQuery,
      variables: { data: { name, color, datasetId } },
      refetchQueries: [
        "getLabelClassesOfDataset",
        { query: getDatasetsQuery },
        { query: datasetLabelClassesQuery, variables: { slug: datasetSlug } },
      ],
    });

    useLabellingStore.setState({ selectedLabelClassId: labelClassId });

    return labelClassId;
  },
  undo: async (labelClassId: string) => {
    await client.mutate({
      mutation: deleteLabelClassQuery,
      variables: {
        where: { id: labelClassId },
      },
      refetchQueries: [
        "getLabelClassesOfDataset",
        { query: getDatasetsQuery },
        { query: datasetLabelClassesQuery, variables: { slug: datasetSlug } },
      ],
    });

    useLabellingStore.setState({
      selectedLabelClassId: selectedLabelClassIdPrevious,
    });

    return labelClassId;
  },
  redo: async (labelClassId: string) => {
    await client.mutate({
      mutation: createLabelClassQuery,
      variables: { data: { name, color, id: labelClassId, datasetId } },
      refetchQueries: [
        "getLabelClassesOfDataset",
        { query: getDatasetsQuery },
        { query: datasetLabelClassesQuery, variables: { slug: datasetSlug } },
      ],
    });

    useLabellingStore.setState({ selectedLabelClassId: labelClassId });

    return labelClassId;
  },
});

export const createNewLabelClassCurry =
  ({
    labelClasses,
    datasetId,
    datasetSlug,
    perform,
    client,
  }: {
    labelClasses: LabelClass[];
    datasetId: string;
    datasetSlug: string;
    perform: any;
    client: ApolloClient<object>;
  }) =>
  async (name: string, selectedLabelClassIdPrevious: string | null) => {
    const newClassColor =
      labelClasses.length < 1
        ? hexColorSequence[0]
        : getNextClassColor(labelClasses[labelClasses.length - 1].color);
    perform(
      createCreateLabelClassEffect(
        {
          name,
          color: newClassColor,
          selectedLabelClassIdPrevious,
          datasetId,
          datasetSlug,
        },
        { client }
      )
    );
  };
