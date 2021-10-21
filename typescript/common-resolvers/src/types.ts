import type {
  Scalars,
  Example as GeneratedExample,
  Image as GeneratedImage,
  Label as GeneratedLabel,
  LabelClass as GeneratedLabelClass,
  Dataset as GeneratedDataset,
  Workspace as GeneratedWorkspace,
  ImageWhereInput,
  LabelClassWhereInput,
  LabelWhereInput,
  UploadTargetHttp,
  UploadTarget,
  DatasetCreateInput,
  LabelClassCreateInput,
  ImageCreateInput,
  User,
  LabelWhereUniqueInput,
  LabelClassWhereUniqueInput,
  DatasetWhereUniqueInput,
  ImageWhereUniqueInput,
  WorkspaceCreateInput,
  WorkspaceWhereInput,
  WorkspaceWhereUniqueInput,
  WorkspaceType,
} from "@labelflow/graphql-types";
import { WorkspacePlan } from "@prisma/client";

type NoUndefinedField<T> = { [P in keyof T]: NonNullable<T[P]> };

export type DbImage = Omit<GeneratedImage, "labels" | "dataset">;
export type DbImageCreateInput = WithCreatedAtAndUpdatedAt<
  Required<NoUndefinedField<Omit<ImageCreateInput, "file" | "externalUrl">>> &
    Pick<ImageCreateInput, "externalUrl">
>;

export type DbLabel = Omit<GeneratedLabel, "labelClass"> & {
  labelClassId: Scalars["ID"] | undefined | null;
};
export type DbLabelCreateInput = WithCreatedAtAndUpdatedAt<DbLabel>;

export type DbLabelClass = Omit<GeneratedLabelClass, "labels" | "dataset"> & {
  datasetId: string;
};
export type DbLabelClassCreateInput = Required<
  NoUndefinedField<
    WithCreatedAtAndUpdatedAt<LabelClassCreateInput & { index: number }>
  >
>;

export type DbExample = GeneratedExample;

export type DbDataset = Omit<
  GeneratedDataset,
  | "images"
  | "imagesAggregates"
  | "labels"
  | "labelsAggregates"
  | "labelClasses"
  | "labelClassesAggregates"
  | "workspace"
> & { workspaceSlug: string };

export type DbWorkspace = Omit<
  GeneratedWorkspace,
  "__typename" | "type" | "datasets" | "memberships" | "plan"
> & { plan: WorkspacePlan };

export type DbWorkspaceWithType = DbWorkspace & { type: WorkspaceType };

export type DbDatasetCreateInput = WithCreatedAtAndUpdatedAt<
  DatasetCreateInput & { slug: string }
>;

export type DbUser = Omit<User, "memberships">;

type PartialWithNullAllowed<T> = { [P in keyof T]?: T[P] | undefined | null };

type WithCreatedAtAndUpdatedAt<T extends {}> = T & {
  createdAt: string;
  updatedAt: string;
};

type ID = string;

type Add<EntityType> = (
  entity: EntityType,
  user?: { id: string }
) => Promise<ID>;
type Count<Where> = (where?: Where) => Promise<number>;
type Delete<EntityWhereUniqueInput> = (
  input: EntityWhereUniqueInput,
  user?: { id: string }
) => Promise<void>;
type Get<EntityType, EntityWhereUniqueInput> = (
  input: EntityWhereUniqueInput,
  user?: { id: string }
) => Promise<EntityType | undefined | null>;

type List<Entity = unknown, Where extends Record<string, any> | null = null> = (
  where?: Where | null,
  skip?: number | null,
  first?: number | null
) => Promise<Entity[]>;
type Update<Entity, EntityWhereUniqueInput> = (
  input: EntityWhereUniqueInput,
  data: PartialWithNullAllowed<Entity>,
  user?: { id: string }
) => Promise<boolean>;

export type Repository = {
  image: {
    add: Add<DbImageCreateInput>;
    count: Count<ImageWhereInput & { user?: { id: string } }>;
    get: Get<DbImage, ImageWhereUniqueInput>;
    list: List<DbImage, ImageWhereInput & { user?: { id: string } }>;
    delete: Delete<ImageWhereUniqueInput>;
  };
  label: {
    add: Add<DbLabelCreateInput>;
    count: Count<LabelWhereInput & { user?: { id: string } }>;
    delete: Delete<LabelWhereUniqueInput>;
    get: Get<DbLabel, LabelWhereUniqueInput>;
    list: List<DbLabel, LabelWhereInput & { user?: { id: string } }>;
    update: Update<DbLabel, LabelWhereUniqueInput>;
  };
  labelClass: {
    add: Add<DbLabelClassCreateInput>;
    count: Count<LabelClassWhereInput & { user?: { id: string } }>;
    delete: Delete<LabelClassWhereUniqueInput>;
    get: Get<DbLabelClass, LabelClassWhereUniqueInput>;
    list: List<DbLabelClass, LabelClassWhereInput & { user?: { id: string } }>;
    update: Update<DbLabelClass, LabelClassWhereUniqueInput>;
  };
  dataset: {
    add: Add<DbDatasetCreateInput>;
    delete: Delete<DatasetWhereUniqueInput>;
    get: Get<DbDataset, DatasetWhereUniqueInput>;
    list: List<DbDataset, { workspaceSlug?: string; user?: { id: string } }>;
    update: Update<DbDataset, DatasetWhereUniqueInput>;
  };
  workspace: {
    add: Add<WorkspaceCreateInput & { slug: string }>;
    get: Get<DbWorkspaceWithType, WorkspaceWhereUniqueInput>;
    list: List<
      DbWorkspaceWithType,
      WorkspaceWhereInput & { user?: { id: string } }
    >;
    update: Update<DbWorkspaceWithType, WorkspaceWhereUniqueInput>;
  };
  upload: {
    getUploadTargetHttp: (
      key: string,
      origin: string
    ) => Promise<UploadTargetHttp> | UploadTargetHttp;
    getUploadTarget: (
      key: string,
      origin: string
    ) => Promise<UploadTarget> | UploadTarget;
    put: (url: string, file: Blob) => Promise<void>;
    get: (url: string, req?: Request) => Promise<ArrayBuffer>;
    delete: (url: string) => Promise<void>;
  };
};

export type Context = {
  repository: Repository;
  user?: { id: string };
  session?: any;
  req?: Request;
};
