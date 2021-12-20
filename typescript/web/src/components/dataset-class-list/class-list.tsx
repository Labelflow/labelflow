import { useMemo, useState, useCallback } from "react";
import { useQuery, gql, useApolloClient } from "@apollo/client";
import {
  Box,
  Text,
  Divider,
  useColorModeValue as mode,
  Heading,
} from "@chakra-ui/react";

import { DragDropContext, Droppable } from "react-beautiful-dnd";
import {
  ClassItem,
  datasetLabelClassesQuery,
  DatasetClassesQueryResult,
} from "./class-item";
import { ClassTableActions } from "./table-actions";
import { ClassTableContent } from "./table-content";
import { DeleteLabelClassModal } from "./delete-class-modal";

const reorderLabelClassMutation = gql`
  mutation reorderLabelClass($id: ID!, $index: Int!) {
    reorderLabelClass(where: { id: $id }, data: { index: $index }) {
      id
    }
  }
`;
const reorder = (list: any[], startIndex: number, endIndex: number) => {
  const result = Array.from(list);
  const [removed] = result.splice(startIndex, 1);
  result.splice(endIndex, 0, removed);
  return result;
};

const addShortcutsToLabelClasses = (labelClasses: any[]) =>
  labelClasses.map((labelClass, index) => ({
    ...labelClass,
    shortcut: index > 9 ? null : `${(index + 1) % 10}`,
  }));

export const ClassesList = ({
  datasetSlug,
  workspaceSlug,
}: {
  datasetSlug: string;
  workspaceSlug: string;
}) => {
  const client = useApolloClient();
  const [editClassId, setEditClassId] = useState<string | null>(null);
  const [deleteClassId, setDeleteClassId] = useState<string | null>(null);
  const [searchText, setSearchText] = useState("");

  const {
    data: datasetResult,
    loading,
    refetch,
    updateQuery,
  } = useQuery<DatasetClassesQueryResult>(datasetLabelClassesQuery, {
    variables: {
      slug: datasetSlug,
      workspaceSlug,
    },
    skip: !datasetSlug,
  });
  const labelClasses = datasetResult?.dataset.labelClasses ?? [];
  const datasetId = datasetResult?.dataset.id;

  const labelClassWithShortcut = useMemo(
    () => addShortcutsToLabelClasses(labelClasses),
    [labelClasses]
  );

  const onDragEnd = useCallback(
    async (result) => {
      // dropped outside the list
      if (!result.destination) {
        return;
      }
      updateQuery((prev) => {
        const labelClassesPrevious = prev?.dataset?.labelClasses;
        return {
          ...prev,
          dataset: {
            ...prev?.dataset,
            labelClasses: addShortcutsToLabelClasses(
              reorder(
                labelClassesPrevious,
                result.source.index,
                result.destination.index
              )
            ),
          },
        };
      });
      await client.mutate({
        mutation: reorderLabelClassMutation,
        variables: { id: result.draggableId, index: result.destination.index },
      });
      refetch();
    },
    [updateQuery]
  );

  return (
    <>
      <DeleteLabelClassModal
        isOpen={deleteClassId != null}
        datasetId={datasetId}
        labelClassId={deleteClassId}
        onClose={() => setDeleteClassId(null)}
      />
      <Box display="flex" flexDirection="column" w="full" p={8}>
        <Heading mb={5}>{`Classes (${labelClassWithShortcut.length})`}</Heading>
        <ClassTableActions
          searchText={searchText}
          setSearchText={setSearchText}
        />
        <ClassTableContent classes={labelClassWithShortcut} />
        <Box
          d="flex"
          flexDirection="column"
          bg={mode("white", "gray.800")}
          borderRadius="lg"
          maxWidth="5xl"
          flexGrow={1}
        >
          <>
            <Text
              margin="2"
              fontWeight="bold"
            >{`${labelClassWithShortcut.length} Classes`}</Text>
            <Divider />
            {!loading && (
              <DragDropContext onDragEnd={onDragEnd}>
                <Droppable droppableId="droppable">
                  {(provided) => (
                    <Box
                      {...provided.droppableProps}
                      ref={provided.innerRef}
                      // style={getListStyle(snapshot.isDraggingOver)}
                    >
                      {labelClassWithShortcut.map(
                        ({ id, name, color, shortcut, index }) => (
                          <ClassItem
                            key={id}
                            id={id}
                            index={index}
                            name={name}
                            color={color}
                            shortcut={shortcut}
                            edit={editClassId === id}
                            datasetSlug={datasetSlug}
                            workspaceSlug={workspaceSlug}
                            onClickEdit={setEditClassId}
                            onClickDelete={setDeleteClassId}
                          />
                        )
                      )}
                      {provided.placeholder}
                    </Box>
                  )}
                </Droppable>
              </DragDropContext>
            )}
          </>
        </Box>
      </Box>
    </>
  );
};
