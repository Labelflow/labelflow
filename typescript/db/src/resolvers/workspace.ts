import { Prisma } from "@prisma/client";

import {
  QueryWorkspaceArgs,
  QueryWorkspacesArgs,
  Workspace,
  Membership,
  MutationCreateWorkspaceArgs,
  MutationUpdateWorkspaceArgs,
  WorkspaceWhereUniqueInput,
} from "@labelflow/graphql-types";

import {
  Context,
  DbWorkspaceWithType,
  forbiddenWorkspaceSlugs,
  isValidWorkspaceName,
  Repository,
} from "@labelflow/common-resolvers";
import slugify from "slugify";
import { getPrismaClient } from "../prisma-client";
import { castObjectNullsToUndefined } from "../repository/utils";

const getWorkspace = async (
  where: WorkspaceWhereUniqueInput,
  repository: Repository,
  user?: { id: string }
): Promise<DbWorkspaceWithType & { __typename: "Workspace" }> => {
  const workspaceFromDb = await repository.workspace.get(where, user);
  if (workspaceFromDb == null) {
    throw new Error(
      `Couldn't find workspace from input "${JSON.stringify(where)}"`
    );
  }
  return { ...workspaceFromDb, __typename: "Workspace" };
};

const workspace = async (
  _: any,
  args: QueryWorkspaceArgs,
  { repository, user }: Context
): Promise<DbWorkspaceWithType> =>
  await getWorkspace(args.where, repository, user);

const workspaces = async (
  _: any,
  args: QueryWorkspacesArgs,
  { repository, user }: Context
): Promise<DbWorkspaceWithType[]> =>
  await repository.workspace.list(
    { user, ...args.where },
    args.skip,
    args.first
  );

const createWorkspace = async (
  _: any,
  args: MutationCreateWorkspaceArgs,
  { repository, user }: Context
): Promise<DbWorkspaceWithType> => {
  if (typeof user?.id !== "string") {
    throw new Error("Couldn't create workspace: No user id");
  }
  const userInDb = await (
    await getPrismaClient()
  ).user.findUnique({ where: { id: user.id } });

  if (userInDb == null) {
    throw new Error(
      `Couldn't create workspace: User with id "${user.id}" doesn't exist in the database`
    );
  }

  const slug = slugify(args.data.name, { lower: true });

  if (slug.length <= 0) {
    throw new Error(`Cannot create a workspace with an empty name.`);
  }

  if (!isValidWorkspaceName(args.data.name)) {
    throw new Error(
      `Cannot create a workspace with the name "${args.data.name}". This name contains invalid characters.`
    );
  }

  if (forbiddenWorkspaceSlugs.includes(slug)) {
    throw new Error(
      `Cannot create a workspace with the slug "${slug}". This slug is reserved.`
    );
  }

  const createdWorkspaceId = await repository.workspace.add(
    {
      id: args.data.id ?? undefined,
      name: args.data.name,
      image: args.data.image ?? undefined,
      slug,
    },
    user
  );

  return await getWorkspace({ id: createdWorkspaceId }, repository, user);
};

const updateWorkspace = async (
  _: any,
  args: MutationUpdateWorkspaceArgs,
  { repository, user }: Context
): Promise<DbWorkspaceWithType> => {
  // We need to get the id of the workspace, to keep track of it even if the slug changes
  const currentWorkspace = await getWorkspace(args.where, repository, user);

  // Update workspace
  await repository.workspace.update(
    castObjectNullsToUndefined(args.where),
    { ...args.data },
    user
  );

  return await getWorkspace({ id: currentWorkspace.id }, repository, user);
};

const memberships = async (parent: Workspace) => {
  return (await (
    await getPrismaClient()
  ).membership.findMany({
    where: { workspaceSlug: parent.slug },
    orderBy: { createdAt: Prisma.SortOrder.asc },
    // needs to be casted to avoid conflicts between enums
  })) as Omit<Membership, "user" | "workspace">[];
};

const datasets = async (parent: Workspace) => {
  return await (
    await getPrismaClient()
  ).dataset.findMany({
    where: { workspaceSlug: parent.slug },
    orderBy: { createdAt: Prisma.SortOrder.asc },
  });
};

export default {
  Query: {
    workspace,
    workspaces,
  },
  Mutation: { createWorkspace, updateWorkspace },
  Workspace: { memberships, datasets },
};
