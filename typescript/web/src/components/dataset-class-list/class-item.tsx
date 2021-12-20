import { useState, useEffect, useCallback, useRef } from "react";
import { gql, useApolloClient } from "@apollo/client";
import {
  Kbd,
  Text,
  IconButton,
  useColorModeValue as mode,
  Flex,
  chakra,
  Input,
  Tooltip,
} from "@chakra-ui/react";
import {
  RiCheckboxBlankCircleFill,
  RiPencilLine,
  RiCheckFill,
  RiCloseFill,
  RiDeleteBin5Line,
} from "react-icons/ri";
import { Draggable } from "react-beautiful-dnd";
import { VscGripper } from "react-icons/vsc";

const CircleIcon = chakra(RiCheckboxBlankCircleFill);
const PenIcon = chakra(RiPencilLine);
const CheckIcon = chakra(RiCheckFill);
const CloseIcon = chakra(RiCloseFill);
const DeleteIcon = chakra(RiDeleteBin5Line);
const DragIcon = chakra(VscGripper);

export type DatasetClassesQueryResult = {
  dataset: {
    id: string;
    name: string;
    labelClasses: {
      id: string;
      index: number;
      name: string;
      color: string;
    }[];
  };
};

type ClassItemProps = {
  id: string;
  name: string;
  index: number;
  color: string;
  shortcut: string | null;
  edit: boolean;
  onClickEdit: (classId: string | null) => void;
  datasetSlug: string;
  workspaceSlug: string;
  onClickDelete: (classId: string | null) => void;
};

export const datasetLabelClassesQuery = gql`
  query getDatasetLabelClasses($slug: String!, $workspaceSlug: String!) {
    dataset(where: { slugs: { slug: $slug, workspaceSlug: $workspaceSlug } }) {
      id
      name
      labelClasses {
        id
        index
        name
        color
      }
    }
  }
`;

const updateLabelClassNameMutation = gql`
  mutation updateLabelClassName($id: ID!, $name: String!) {
    updateLabelClass(where: { id: $id }, data: { name: $name }) {
      id
      name
    }
  }
`;

export const ClassItem = ({
  id,
  name,
  index,
  color,
  shortcut,
  edit,
  onClickEdit,
  datasetSlug,
  workspaceSlug,
  onClickDelete,
}: ClassItemProps) => {
  const [editName, setEditName] = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);

  const isEditing = edit && editName !== null;

  useEffect(() => {
    if (inputRef.current && isEditing) {
      inputRef.current.focus();
    }
  }, [isEditing]);

  const client = useApolloClient();
  useEffect(() => {
    if (edit) {
      setEditName(name);
    } else {
      setEditName(null);
    }
  }, [edit]);

  const updateLabelClassNameWithOptimistic = useCallback(() => {
    onClickEdit(null);
    client.mutate({
      mutation: updateLabelClassNameMutation,
      variables: { id, name: editName },
      optimisticResponse: {
        updateLabelClass: {
          id,
          name: editName,
          color,
          __typeName: "LabelClass",
        },
      },
      update: (cache, { data }) => {
        if (data != null) {
          const { updateLabelClass } = data;
          const datasetCacheResult = cache.readQuery<DatasetClassesQueryResult>(
            {
              query: datasetLabelClassesQuery,
              variables: { slug: datasetSlug, workspaceSlug },
            }
          );
          if (datasetCacheResult?.dataset == null) {
            throw new Error(`Missing dataset with slug ${datasetSlug}`);
          }
          const { dataset } = datasetCacheResult;
          const updatedDataset = {
            ...dataset,
            labelClasses: dataset.labelClasses.map((labelClass) =>
              labelClass.id !== id ? labelClass : { ...updateLabelClass }
            ),
          };
          cache.writeQuery({
            query: datasetLabelClassesQuery,
            variables: { slug: datasetSlug, workspaceSlug },
            data: { dataset: updatedDataset },
          });
        } else {
          throw new Error(
            "Received null data in update label class name function"
          );
        }
      },
    });
  }, [editName, id, datasetSlug, onClickEdit]);

  return (
    <Draggable key={id} draggableId={id} index={index}>
      {(provided, snapshot) => (
        <div ref={provided.innerRef} {...provided.draggableProps}>
          <Flex
            alignItems="center"
            height="10"
            bg={mode("white", "gray.800")}
            borderRadius="lg"
            boxShadow={snapshot.isDragging ? "lg" : undefined}
          >
            <div {...provided.dragHandleProps}>
              <Tooltip
                placement="bottom"
                openDelay={300}
                label={`Reorder class ${name}`}
                aria-label={`Reorder class ${name}`}
              >
                <IconButton
                  variant="ghost"
                  aria-label="Drag"
                  alignItems="center"
                  justifyContent="center"
                  ml="1"
                  icon={
                    <DragIcon
                      color={mode("gray.600", "gray.400")}
                      h="5"
                      flexShrink={0}
                      flexGrow={0}
                    />
                  }
                  h="8"
                  w="8"
                  minWidth="8"
                />
              </Tooltip>
            </div>
            <CircleIcon
              flexShrink={0}
              flexGrow={0}
              color={color}
              fontSize="2xl"
              ml="2"
              mr="2"
            />

            <Input
              ref={inputRef}
              display={isEditing ? "block" : "none"}
              aria-label="Class name input"
              variant="flushed"
              flexGrow={1}
              isTruncated
              value={editName || ""}
              onChange={(e) => setEditName(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === "Enter" && editName !== "") {
                  updateLabelClassNameWithOptimistic();
                }
              }}
            />
            <Text
              display={!isEditing ? "block" : "none"}
              flexGrow={1}
              isTruncated
            >
              {name}
            </Text>

            {shortcut && (
              <Kbd flexShrink={0} flexGrow={0} justifyContent="center" mr="1">
                {shortcut}
              </Kbd>
            )}

            {isEditing ? (
              <>
                <Tooltip
                  placement="bottom"
                  openDelay={300}
                  label="Cancel"
                  aria-label="Cancel"
                >
                  <IconButton
                    variant="ghost"
                    aria-label="Cancel"
                    icon={
                      <CloseIcon
                        flexShrink={0}
                        flexGrow={0}
                        color={mode("gray.600", "gray.400")}
                      />
                    }
                    h="8"
                    w="8"
                    mr="1"
                    minWidth="8"
                    onClick={() => onClickEdit(null)}
                  />
                </Tooltip>
                <Tooltip
                  placement="bottom"
                  openDelay={300}
                  label="Save"
                  aria-label="Save"
                >
                  <IconButton
                    variant="ghost"
                    aria-label="Save"
                    icon={
                      <CheckIcon
                        flexShrink={0}
                        flexGrow={0}
                        color={mode("gray.600", "gray.400")}
                      />
                    }
                    h="8"
                    w="8"
                    mr="1"
                    minWidth="8"
                    onClick={updateLabelClassNameWithOptimistic}
                    disabled={editName === ""}
                  />
                </Tooltip>
              </>
            ) : (
              <>
                <Tooltip
                  placement="bottom"
                  openDelay={300}
                  label={`Edit name of class ${name}`}
                >
                  <IconButton
                    variant="ghost"
                    aria-label={`Edit class ${name} name`}
                    icon={
                      <PenIcon
                        flexShrink={0}
                        flexGrow={0}
                        color={mode("gray.600", "gray.400")}
                      />
                    }
                    h="8"
                    w="8"
                    mr="1"
                    minWidth="8"
                    onClick={() => onClickEdit(id)}
                  />
                </Tooltip>
                <Tooltip
                  placement="bottom"
                  openDelay={300}
                  label={`Delete class ${name}`}
                >
                  <IconButton
                    variant="ghost"
                    aria-label="Delete class"
                    icon={
                      <DeleteIcon
                        flexShrink={0}
                        flexGrow={0}
                        color={mode("gray.600", "gray.400")}
                      />
                    }
                    h="8"
                    w="8"
                    mr="1"
                    minWidth="8"
                    onClick={() => onClickDelete(id)}
                  />
                </Tooltip>
              </>
            )}
          </Flex>
        </div>
      )}
    </Draggable>
  );
};
