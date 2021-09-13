import { gql } from "@apollo/client";
import { v4 as uuidV4 } from "uuid";
import { User } from "@labelflow/graphql-types";
import { prisma } from "../../repository";
import { client, user as loggedInUser } from "../../dev/apollo-client";

// @ts-ignore
fetch.disableFetchMocks();

const testUser1Id = uuidV4();
const testUser2Id = uuidV4();
const testUser3Id = uuidV4();
const testUser4Id = uuidV4();

beforeEach(async () => {
  await prisma.membership.deleteMany({});
  await prisma.workspace.deleteMany({});
  return await prisma.user.deleteMany({});
});

describe("users query", () => {
  it("returns an empty array when there aren't any", async () => {
    const { data } = await client.query({
      query: gql`
        {
          users {
            id
          }
        }
      `,
      fetchPolicy: "no-cache",
    });

    expect(data.users).toEqual([]);
  });

  it("returns the already created users", async () => {
    await prisma.user.create({
      data: { id: testUser1Id, name: "test-user-1" },
    });
    await prisma.user.create({
      data: { id: testUser2Id, name: "test-user-2" },
    });
    await prisma.user.create({
      data: { id: testUser3Id, name: "test-user-3" },
    });
    await prisma.user.create({
      data: { id: testUser4Id, name: "test-user-4" },
    });

    const { data } = await client.query<{
      users: Pick<User, "id" | "name">[];
    }>({
      query: gql`
        {
          users {
            id
            name
          }
        }
      `,
      fetchPolicy: "no-cache",
    });

    expect(data.users.map((user) => user.name)).toEqual([
      "test-user-1",
      "test-user-2",
      "test-user-3",
      "test-user-4",
    ]);
  });

  it("can skip results", async () => {
    await prisma.user.create({
      data: { id: testUser1Id, name: "test-user-1" },
    });
    await prisma.user.create({
      data: { id: testUser2Id, name: "test-user-2" },
    });
    await prisma.user.create({
      data: { id: testUser3Id, name: "test-user-3" },
    });
    await prisma.user.create({
      data: { id: testUser4Id, name: "test-user-4" },
    });

    const { data } = await client.query<{
      users: Pick<User, "id" | "name">[];
    }>({
      query: gql`
        {
          users(skip: 1) {
            id
            name
          }
        }
      `,
      fetchPolicy: "no-cache",
    });

    expect(data.users.map((user) => user.name)).toEqual([
      "test-user-2",
      "test-user-3",
      "test-user-4",
    ]);
  });

  it("can limit the number of results", async () => {
    await prisma.user.create({
      data: { id: testUser1Id, name: "test-user-1" },
    });
    await prisma.user.create({
      data: { id: testUser2Id, name: "test-user-2" },
    });
    await prisma.user.create({
      data: { id: testUser3Id, name: "test-user-3" },
    });
    await prisma.user.create({
      data: { id: testUser4Id, name: "test-user-4" },
    });

    const { data } = await client.query<{
      users: Pick<User, "id" | "name">[];
    }>({
      query: gql`
        {
          users(first: 3) {
            id
            name
          }
        }
      `,
      fetchPolicy: "no-cache",
    });

    expect(data.users.map((user) => user.name)).toEqual([
      "test-user-1",
      "test-user-2",
      "test-user-3",
    ]);
  });

  it("can limit the number of results and skip some", async () => {
    await prisma.user.create({
      data: { id: testUser1Id, name: "test-user-1" },
    });
    await prisma.user.create({
      data: { id: testUser2Id, name: "test-user-2" },
    });
    await prisma.user.create({
      data: { id: testUser3Id, name: "test-user-3" },
    });
    await prisma.user.create({
      data: { id: testUser4Id, name: "test-user-4" },
    });

    const { data } = await client.query<{
      users: Pick<User, "id" | "name">[];
    }>({
      query: gql`
        {
          users(skip: 1, first: 2) {
            id
            name
          }
        }
      `,
      fetchPolicy: "no-cache",
    });

    expect(data.users.map((user) => user.name)).toEqual([
      "test-user-2",
      "test-user-3",
    ]);
  });
});

describe("user query", () => {
  const queryUser = async (id: string) =>
    await client.query<{
      user: Pick<User, "id" | "name">;
    }>({
      query: gql`
        query user($id: ID!) {
          user(where: { id: $id }) {
            id
            name
          }
        }
      `,
      variables: { id },
      fetchPolicy: "no-cache",
    });

  it("returns the user corresponding to the given id", async () => {
    await prisma.user.create({
      data: { id: testUser1Id, name: "test-user-1" },
    });

    const { data } = await queryUser(testUser1Id);

    expect(data.user.name).toEqual("test-user-1");
  });

  it("throws if the provided id doesn't match any user", async () => {
    const idOfAnUserThaDoesNotExist = uuidV4();

    await expect(() => queryUser(idOfAnUserThaDoesNotExist)).rejects.toThrow(
      `Couldn't find an user with id: "${idOfAnUserThaDoesNotExist}"`
    );
  });
});

describe("updateUser mutation", () => {
  const updateUser = async ({
    id,
    name,
    image,
  }: {
    id: string;
    name?: string;
    image?: string;
  }) =>
    await client.mutate<{
      updateUser: Pick<User, "id" | "name">;
    }>({
      mutation: gql`
        mutation updateUser($id: ID!, $data: UserUpdateInput!) {
          updateUser(where: { id: $id }, data: $data) {
            id
            name
            image
          }
        }
      `,
      variables: { id, data: { name, image } },
      fetchPolicy: "no-cache",
    });

  it("can change the user name", async () => {
    await prisma.user.create({
      data: { id: testUser1Id, name: "test-user-1" },
    });

    const { data } = await updateUser({
      id: testUser1Id,
      name: "New name",
    });

    expect(data?.updateUser.name).toEqual("New name");
  });

  it("can change the user image", async () => {
    await prisma.user.create({
      data: { id: testUser1Id, name: "test-user-1" },
    });

    const { data } = await updateUser({
      id: testUser1Id,
      name: "New image",
    });

    expect(data?.updateUser.name).toEqual("New image");
  });

  it("throws if the membership to update doesn't exist", async () => {
    const idOfAnUserThaDoesNotExist = uuidV4();

    await expect(() =>
      updateUser({
        id: idOfAnUserThaDoesNotExist,
        name: "This will fail anyway",
      })
    ).rejects.toThrow();
  });
});

describe("nested resolvers", () => {
  const createWorkspace = async (
    data?: Partial<MutationCreateWorkspaceArgs["data"]>
  ) => {
    return await client.mutate<{
      createWorkspace: Pick<
        Workspace,
        "id" | "name" | "slug" | "plan" | "type"
      >;
    }>({
      mutation: gql`
        mutation createWorkspace($data: WorkspaceCreateInput!) {
          createWorkspace(data: $data) {
            id
            name
            slug
            plan
            type
          }
        }
      `,
      variables: { data: { ...data, name: data?.name ?? "test" } },
    });
  };

  it("can return user's memberships", async () => {
    await prisma.user.create({
      data: { id: testUser1Id, name: "test-user-1" },
    });
    loggedInUser.id = testUser1Id;
    await createWorkspace({ name: "test" });

    const { data } = await client.query<{
      user: Pick<User, "id" | "memberships">;
    }>({
      query: gql`
        query user($id: ID!) {
          user(where: { id: $id }) {
            id
            memberships {
              id
              workspace {
                id
              }
            }
          }
        }
      `,
      variables: { id: testUser1Id },
      fetchPolicy: "no-cache",
    });

    expect(data.user.memberships.length).toEqual(1);
  });
});