import { gql, useQuery, useApolloClient, ApolloError } from "@apollo/client";
import { Workspace } from "@labelflow/graphql-types";
import { useToast } from "@chakra-ui/react";
import { useRouter } from "next/router";
import { useState, useCallback, useMemo } from "react";
import { startCase } from "lodash/fp";
import { useQueryParam } from "use-query-params";

import { WorkspaceMenu } from "./workspace-menu";
import { WorkspaceItem } from "./workspace-menu/workspace-selection-popover";
import { BoolParam } from "../../utils/query-param-bool";

const getWorkspacesQuery = gql`
  query getWorkspaces {
    workspaces {
      id
      name
      slug
    }
  }
`;

const createWorkspacesQuery = gql`
  mutation createWorkspace($name: String!) {
    createWorkspace(data: { name: $name }) {
      id
      name
      slug
    }
  }
`;

export const WorkspaceSwitcher = () => {
  const router = useRouter();
  const [, setSigninModalOpen] = useQueryParam("modal-signin", BoolParam);

  const workspaceSlug = router?.query.workspaceSlug as string;

  const client = useApolloClient();

  const { data: getWorkspacesData, previousData: getWorkspacesPreviousData } =
    useQuery(getWorkspacesQuery, {
      variables: { workspaceSlug },
    });

  const workspaces: (Workspace & { src?: string })[] = [
    { id: "local", slug: "local", name: "Local", src: null },
    ...(getWorkspacesData?.workspaces ??
      getWorkspacesPreviousData?.workspaces ??
      []),
  ];

  const [isOpen, setIsOpen] = useState(false);

  const toast = useToast();

  const selectedWorkspace = useMemo(() => {
    if (workspaceSlug == null) {
      return null;
    }
    if (workspaces == null) {
      return {
        id: "local",
        slug: workspaceSlug,
        name: startCase(workspaceSlug),
        src: null,
      };
    }
    return workspaces.find(({ slug }) => slug === workspaceSlug);
  }, [workspaceSlug, workspaces]);

  const setSelectedWorkspace = useCallback((workspace: WorkspaceItem) => {
    const slug = workspace?.slug;
    console.log(`Switch to workspace ${slug ?? "unknown"}`);
    if (slug !== null) {
      router.push(`/${slug}`);
    }
  }, []);

  const createNewWorkspace = useCallback(
    async (name: string) => {
      try {
        const { data, errors } = await client.mutate({
          mutation: createWorkspacesQuery,
          variables: { name },
        });
        const slug = data?.createWorkspace?.slug;

        if (slug !== null) {
          router.push(`/${slug}`);
        } else {
          toast({
            title: "Could not create workspace",
            description: errors?.[0],
            isClosable: true,
            status: "error",
            position: "bottom-right",
            duration: 10000,
          });
        }
      } catch (error: any) {
        if (error instanceof ApolloError) {
          toast({
            title: "Needs to be signed in",
            description:
              "Only signed-in users can to create and share Workspaces online, please sign in.",
            isClosable: true,
            status: "info",
            position: "bottom-right",
            duration: 10000,
          });
          setSigninModalOpen(true, "replaceIn");
        } else {
          toast({
            title: "Could not create workspace",
            description: error?.message ?? error,
            isClosable: true,
            status: "error",
            position: "bottom-right",
            duration: 10000,
          });
        }
      }
    },
    [client, toast]
  );

  if (workspaces == null) {
    return null;
  }

  if (selectedWorkspace == null) {
    return (
      <WorkspaceMenu
        isOpen={isOpen}
        setIsOpen={setIsOpen}
        workspaces={workspaces}
        onSelectedWorkspaceChange={(workspace: WorkspaceItem) =>
          setSelectedWorkspace(workspace)
        }
        createNewWorkspace={createNewWorkspace}
        selectedWorkspace={null}
      />
    );
  }

  return (
    <WorkspaceMenu
      isOpen={isOpen}
      setIsOpen={setIsOpen}
      workspaces={workspaces}
      onSelectedWorkspaceChange={(workspace: WorkspaceItem) =>
        setSelectedWorkspace(workspace)
      }
      createNewWorkspace={createNewWorkspace}
      selectedWorkspace={selectedWorkspace}
    />
  );
};