import { Transaction, Dexie } from "dexie";

type DbVersion = {
  name: string;
  version: number;
  stores: {
    [key: string]: string;
  };
  upgrade: ((t: Transaction) => Promise<void>) | ((t: Transaction) => void);
};

// https://dexie.org/docs/Version/Version.stores()
// First key is set to be the primary key and has to be unique
export default [
  {
    name: "20210527-1424-first version",
    version: 0.1,
    stores: {
      example: "id,createdAt,updatedAt,name",
      image:
        "id,createdAt,updatedAt,url,name,path,mimetype,width,height,fileId,projectId",
      label:
        "id,createdAt,updatedAt,imageId,x,y,height,width,labelClassId,geometry",
      labelClass: "id,createdAt,updatedAt,name,color,projectId",
      project: "id,createdAt,updatedAt,&name",
    },
    upgrade: () => {},
  },
  {
    name: "20210706-1136-second version",
    version: 0.2,
    stores: {},
    upgrade: async (transaction: Transaction): Promise<void> => {
      const defaultProjectId = "f21b46fb-d6c3-4fe3-aa5c-0b38ff1af7df";
      await Dexie.waitFor(
        // @ts-ignore
        transaction.db.project.add({
          // Uuid generated by hand for the sake of the migration
          // DON'T USE IT ANYWHERE ELSE
          id: defaultProjectId,
          createdAt: "2021-07-06T11:36:00.000Z",
          updatedAt: "2021-07-06T11:36:00.000Z",
          name: "labelflow default project",
        })
      );
      await Dexie.waitFor(
        // @ts-ignore
        transaction.image.toCollection().modify((image) => {
          // eslint-disable-next-line no-param-reassign
          image.projectId = defaultProjectId;
        })
      );
      // @ts-ignore
      return transaction.labelClass.toCollection().modify((labelClass) => {
        // eslint-disable-next-line no-param-reassign
        labelClass.projectId = defaultProjectId;
      });
    },
  },
] as Array<DbVersion>;
