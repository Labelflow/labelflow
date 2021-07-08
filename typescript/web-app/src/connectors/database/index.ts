import Dexie from "dexie";
import versions from "./versions";
import type {
  Scalars,
  Example as GeneratedExample,
  Image as GeneratedImage,
  Label as GeneratedLabel,
  LabelClass as GeneratedLabelClass,
  Project as GeneratedProject,
} from "../../graphql-types.generated";

export type DbFile = {
  id: string;
  blob: Blob;
};

export type DbImage =
  | (Omit<GeneratedImage, "url" | "labels"> & {
      fileId: Scalars["ID"];
    })
  | Omit<GeneratedImage, "labels">;

export type DbLabel = Omit<GeneratedLabel, "labelClass"> & {
  labelClassId: Scalars["ID"] | undefined | null;
};

export type DbLabelClass = Omit<GeneratedLabelClass, "labels">;

export type DbExample = GeneratedExample;

export type DbProject = Omit<
  GeneratedProject,
  | "images"
  | "imagesCount"
  | "labelClasses"
  | "labelClassesCount"
  | "labelsCount"
>;

interface Database extends Dexie {
  example: Dexie.Table<DbExample, string>;
  image: Dexie.Table<DbImage, string>;
  file: Dexie.Table<DbFile, string>;
  label: Dexie.Table<DbLabel, string>;
  labelClass: Dexie.Table<DbLabelClass, string>;
  project: Dexie.Table<DbProject, string>;
}

// eslint-disable-next-line import/no-mutable-exports
export let db: Database;

export const resetDatabase = () => {
  console.log("Initializing database");
  if (db) {
    try {
      db.close();
    } catch (e) {
      console.log("Could not close existing database");
    }
  }
  db = new Dexie("labelflow_local") as Database;
  versions.map(({ version, stores, upgrade }) =>
    db.version(version).stores(stores).upgrade(upgrade)
  );

  db.on("populate", () => {
    const defaultProjectId = "f21b46fb-d6c3-4fe3-aa5c-0b38ff1af7df";
    // @ts-ignore
    db.project.put({
      // Uuid generated by hand for the sake of the migration
      // DON'T USE IT ANYWHERE ELSE
      id: defaultProjectId,
      createdAt: "2021-07-06T11:36:00.000Z",
      updatedAt: "2021-07-06T11:36:00.000Z",
      name: "labelflow default project",
    });
  });
};

resetDatabase();
