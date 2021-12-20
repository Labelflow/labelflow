import mime from "mime-types";
import JSZip from "jszip";

export enum ValidMimeTypeCategory {
  image = "image",
  archive = "archive",
  json = "json",
}

export const validMimeTypes: Record<ValidMimeTypeCategory, string[]> = {
  image: ["image/jpeg", "image/png", "image/bmp"],
  archive: ["application/zip"],
  json: ["application/json"],
};

export const validMimeTypesFlat: string[] = Object.values(
  validMimeTypes
).reduce<string[]>(
  (mimeTypesFlat, mimeTypeList) =>
    mimeTypeList.reduce<string[]>((mimeTypesFlatSub, mimeTypeString) => {
      mimeTypesFlatSub.push(mimeTypeString);
      return mimeTypesFlatSub;
    }, mimeTypesFlat),
  [] as string[]
);

export function getValidMimeTypeCategoriesList(
  categories: ValidMimeTypeCategory[] | undefined = undefined
): string[] {
  return categories === undefined
    ? validMimeTypesFlat
    : Object.values(categories).reduce<string[]>(
        (mimeTypesFlat, mimeTypeCategory) =>
          validMimeTypes[mimeTypeCategory].reduce<string[]>(
            (mimeTypesFlatSub, mimeTypeString) => {
              mimeTypesFlatSub.push(mimeTypeString);
              return mimeTypesFlatSub;
            },
            mimeTypesFlat
          ),
        [] as string[]
      );
}

export function getValidMimeTypeCategoriesListString(
  categories: ValidMimeTypeCategory[] | undefined = undefined
): string {
  return getValidMimeTypeCategoriesList(categories).join(", ");
}

export const validMimeTypesFlatString = getValidMimeTypeCategoriesListString();

export function isFilePathOfValidMimeTypeCategory(
  filePath: string,
  category: ValidMimeTypeCategory
): boolean {
  const filePathMimeType = mime.lookup(filePath);
  return filePathMimeType === false
    ? false
    : validMimeTypes[category].includes(filePathMimeType as string);
}

// TODO implement in a way that works both for front and back
/*
export async function isStreamOfValidMimeTypeCategory(
  stream: NodeJS.ReadableStream,
  category: ValidMimeTypeCategory
): Promise<boolean> {
  return true;
  const { mime: streamMimeType } = await fileTypeFromStream(stream as Readable);
  return streamMimeType
    ? validMimeTypes[category].includes(streamMimeType)
    : false;
}
*/

// TODO also check binary mime type markers
export async function isJSZipObjectOfValidMimeTypeCategory(
  filePath: string,
  jszipObject: JSZip.JSZipObject,
  category: ValidMimeTypeCategory
): Promise<boolean> {
  return isFilePathOfValidMimeTypeCategory(filePath, category);
  /* && (await isStreamOfValidMimeTypeCategory(jszipObject.nodeStream(), category)) */
}