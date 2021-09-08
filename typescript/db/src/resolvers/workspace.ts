import slugify from "slugify";
import { Prisma } from "@prisma/client";

import {
  QueryWorkspaceArgs,
  QueryWorkspacesArgs,
  WorkspaceType,
  WorkspacePlan,
  Workspace,
  MutationCreateWorkspaceArgs,
  MutationUpdateWorkspaceArgs,
} from "@labelflow/graphql-types";

import { Context } from "@labelflow/common-resolvers";
import { prisma } from "../repository";

type DbWorkspacePlan = NonNullable<
  Prisma.PromiseReturnType<typeof prisma.workspace.findUnique>
>["plan"];

type DbWorkspace = Omit<
  Workspace,
  "__typename" | "type" | "datasets" | "memberships" | "plan"
> & { plan: DbWorkspacePlan };

type DbWorkspaceWithType = DbWorkspace & { type: WorkspaceType };

const addTypeToWorkspace = (
  workspaceWithoutType: Omit<DbWorkspace, "type">
): DbWorkspaceWithType => ({
  ...workspaceWithoutType,
  type: WorkspaceType.Online,
});

const workspace = async (
  _: any,
  args: QueryWorkspaceArgs
): Promise<DbWorkspaceWithType> => {
  const workspaceFromDb = await prisma.workspace.findUnique({
    where: args.where,
  });

  if (workspaceFromDb == null) {
    throw new Error(`Couldn't find a workspace with id: ${args.where.id}`);
  }
  return addTypeToWorkspace(workspaceFromDb);
};

const workspaces = async (
  _: any,
  args: QueryWorkspacesArgs
): Promise<DbWorkspaceWithType[]> => {
  console.log("hello");
  const workspacesFromDb = await prisma.workspace.findMany({
    skip: args.skip ?? undefined,
    take: args.first ?? undefined,
    orderBy: { createdAt: Prisma.SortOrder.asc },
  });

  return workspacesFromDb.map(addTypeToWorkspace);
};

const createWorkspace = async (
  _: any,
  args: MutationCreateWorkspaceArgs,
  { user }: Context
): Promise<DbWorkspaceWithType> => {
  if (typeof user?.id !== "string") {
    throw new Error("Couldn't create workspace: No user id");
  }
  const userInDb = await prisma.user.findUnique({ where: { id: user.id } });

  if (userInDb == null) {
    throw new Error(
      `Couldn't create workspace: User with id "${user.id}"" doesn't exist in the database`
    );
  }

  const createdWorkspace = await prisma.workspace.create({
    data: {
      id: args.data.id ?? undefined,
      name: args.data.name,
      slug: slugify(args.data.name, { lower: true }),
      plan: WorkspacePlan.Community,
      memberships: { create: { userId: user?.id, role: "Admin" } },
    },
  });

  return addTypeToWorkspace(createdWorkspace);
};

const updateWorkspace = async (
  _: any,
  args: MutationUpdateWorkspaceArgs
): Promise<DbWorkspaceWithType> => {
  const dataWithSlug =
    typeof args.data.name === "string"
      ? {
          ...args.data,
          name: args.data.name ?? undefined,
          slug: slugify(args.data.name, { lower: true }),
        }
      : { ...args.data, name: args.data.name ?? undefined };

  const updatedWorkspace = await prisma.workspace.update({
    where: args.where,
    data: dataWithSlug,
  });

  return addTypeToWorkspace(updatedWorkspace);
};

export default {
  Query: {
    workspace,
    workspaces,
  },
  Mutation: { createWorkspace, updateWorkspace },
};
