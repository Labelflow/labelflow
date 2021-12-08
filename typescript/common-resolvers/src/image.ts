import { v4 as uuidv4 } from "uuid";
import "isomorphic-fetch";

import type {
  ImageCreateInput,
  MutationCreateImageArgs,
  MutationUpdateImageArgs,
  QueryImageArgs,
  QueryImagesArgs,
  MutationDeleteImageArgs,
} from "@labelflow/graphql-types";
import mime from "mime-types";
import { Context, DbImage, Repository, DbImageCreateInput } from "./types";
import { throwIfResolvesToNil } from "./utils/throw-if-resolves-to-nil";
import { getOrigin } from "./utils/get-origin";

// Mutations
const getImageFileKey = (
  imageId: string,
  workspaceId: string,
  datasetId: string,
  mimetype: string
) => `${workspaceId}/${datasetId}/${imageId}.${mime.extension(mimetype)}`;

const getImageName = ({
  externalUrl,
  finalUrl,
  name,
}: {
  externalUrl?: string | null;
  finalUrl?: string | null;
  name?: string | null;
}): string => {
  const nameBase =
    name ??
    externalUrl?.substring(
      externalUrl?.lastIndexOf("/") + 1,
      externalUrl?.indexOf("?")
    ) ??
    finalUrl!.substring(finalUrl!.lastIndexOf("/") + 1, finalUrl!.indexOf("?"));
  return nameBase.replace(/\.[^/.]+$/, "");
};

/**
 * Very important function, which processes images (download from external URL if needed, probe metadata, create and upload thumbnails, etc.)
 * @param data ImageCreateInput
 * @param repository
 * @param req
 * @returns
 */
export const getImageEntityFromMutationArgs = async (
  data: ImageCreateInput,
  repository: Repository,
  user?: { id: string },
  req?: Request
) => {
  const {
    file,
    id,
    name,
    height,
    width,
    mimetype,
    path,
    url,
    externalUrl,
    datasetId,
    thumbnail20Url,
    thumbnail50Url,
    thumbnail100Url,
    thumbnail200Url,
    thumbnail500Url,
  } = data;
  const { workspaceSlug } = (await repository.dataset.get(
    { id: datasetId },
    user
  )) as { workspaceSlug: string };
  const { id: workspaceId } = (await repository.workspace.get(
    {
      slug: workspaceSlug,
    },
    user
  )) as { id: string };

  const now = data?.createdAt ?? new Date().toISOString();
  const imageId = id ?? uuidv4();
  let finalUrl: string | undefined;

  let thumbnailsUrls: { [key: string]: string } = {};
  if (thumbnail20Url) thumbnailsUrls.thumbnail20Url = thumbnail20Url;
  if (thumbnail50Url) thumbnailsUrls.thumbnail50Url = thumbnail50Url;
  if (thumbnail100Url) thumbnailsUrls.thumbnail100Url = thumbnail100Url;
  if (thumbnail200Url) thumbnailsUrls.thumbnail200Url = thumbnail200Url;
  if (thumbnail500Url) thumbnailsUrls.thumbnail500Url = thumbnail500Url;

  if (!file && !externalUrl && url) {
    // No File Upload
    finalUrl = url;
  }

  const origin = getOrigin(req);
  if (!file && externalUrl && !url) {
    // External file based upload

    const headers = new Headers();
    headers.set("Accept", "image/tiff,image/jpeg,image/png,image/*,*/*;q=0.8");
    headers.set("Sec-Fetch-Dest", "image");
    if ((req?.headers as any)?.cookie) {
      headers.set("Cookie", (req?.headers as any)?.cookie);
    }

    const fetchResult = await fetch(externalUrl, {
      method: "GET",
      mode: "cors",
      headers,
      credentials: "omit",
    });

    if (fetchResult.status !== 200) {
      throw new Error(
        `While transferring image could not fetch image at url ${externalUrl} properly, code ${fetchResult.status}`
      );
    }

    const blob = await fetchResult.blob();
    const uploadTarget = await repository.upload.getUploadTargetHttp(
      getImageFileKey(imageId, workspaceId, datasetId, blob.type),
      origin
    );

    // eslint-disable-next-line no-underscore-dangle
    if (uploadTarget.__typename !== "UploadTargetHttp") {
      throw new Error(
        "This Server does not support file upload. You can create images by providing a `file` directly in the `createImage` mutation"
      );
    }

    await repository.upload.put(uploadTarget.uploadUrl, blob, req);

    finalUrl = uploadTarget.downloadUrl;
  }

  if (file && !externalUrl && !url) {
    // File Content based upload
    const uploadTarget = await repository.upload.getUploadTargetHttp(
      getImageFileKey(imageId, workspaceId, datasetId, file.type),
      origin
    );

    // eslint-disable-next-line no-underscore-dangle
    if (uploadTarget.__typename !== "UploadTargetHttp") {
      throw new Error(
        "This Server does not support file upload. You can create images by providing a `file` directly in the `createImage` mutation"
      );
    }

    await repository.upload.put(uploadTarget.uploadUrl, file, req);

    finalUrl = uploadTarget.downloadUrl;
  }

  if (data.noThumbnails) {
    // Do not generate or store thumbnails on server, use either the thumbnails url provided above, or use the full size image as thumbnails
    thumbnailsUrls = {
      thumbnail20Url: finalUrl!,
      thumbnail50Url: finalUrl!,
      thumbnail100Url: finalUrl!,
      thumbnail200Url: finalUrl!,
      thumbnail500Url: finalUrl!,
      ...thumbnailsUrls,
    };
  }

  const downloadUrlPrefix = (
    await repository.upload.getUploadTargetHttp("", origin)
  ).downloadUrl;
  // Probe the file to get its dimensions and mimetype if not provided
  const imageMetaData = await repository.imageProcessing.processImage(
    {
      ...thumbnailsUrls,
      id: imageId,
      width,
      height,
      mimetype,
      url: finalUrl!,
    },
    (fromUrl: string) => repository.upload.get(fromUrl, req),
    async (targetDownloadUrl: string, blob: Blob) => {
      const key = targetDownloadUrl.substring(downloadUrlPrefix.length);
      const toUrl = (await repository.upload.getUploadTargetHttp(key, origin))
        .uploadUrl;
      await repository.upload.put(toUrl, blob, req);
    },
    repository.image.update,
    user
  );

  const newImageEntity: DbImageCreateInput = {
    datasetId,
    createdAt: now,
    updatedAt: now,
    id: imageId,
    url: finalUrl!,
    externalUrl,
    path: path ?? externalUrl ?? finalUrl!,
    name: getImageName({ externalUrl, finalUrl, name }),
    ...imageMetaData,
  };

  return newImageEntity;
};

const getImageById = async (
  id: string,
  repository: Repository,
  user?: { id: string }
): Promise<DbImage> => {
  return await throwIfResolvesToNil(
    `No image with id "${id}"`,
    repository.image.get
  )({ id }, user);
};

// Queries
const labelsResolver = async (
  { id }: DbImage,
  _args: any,
  { repository, user }: Context
) => {
  return await repository.label.list({ imageId: id, user });
};

const thumbnailResolver =
  (size: 20 | 50 | 100 | 200 | 500) =>
  async (dbImage: DbImage): Promise<string> => {
    return (
      (dbImage as unknown as { [key: string]: string })[
        `thumbnail${size}Url`
      ] ??
      dbImage.url ??
      dbImage.externalUrl
    );
  };

const image = async (
  _: any,
  args: QueryImageArgs,
  { repository, user }: Context
) => {
  return await getImageById(args?.where?.id, repository, user);
};

const images = async (
  _: any,
  args: QueryImagesArgs,
  { repository, user }: Context
) => {
  return await repository.image.list(
    { ...args?.where, user },
    args?.skip,
    args?.first
  );
};

// Mutations
const createImage = async (
  _: any,
  args: MutationCreateImageArgs,
  { repository, req, user }: Context
): Promise<DbImage> => {
  const { file, url, externalUrl, datasetId } = args.data;

  // Since we don't have any constraint checks with Dexie
  // we need to ensure that the datasetId matches some
  // entity before being able to continue.
  await throwIfResolvesToNil(
    `The dataset id ${datasetId} doesn't exist.`,
    repository.dataset.get
  )({ id: datasetId }, user);

  if (
    !(
      (!file && !externalUrl && url) ||
      (!file && externalUrl && !url) ||
      (file && !externalUrl && !url)
    )
  ) {
    throw new Error(
      "Image creation upload must include either a `file` field of type `Upload`, or a `url` field of type `String`, or a `externalUrl` field of type `String`"
    );
  }

  const newImageEntity = await getImageEntityFromMutationArgs(
    args.data,
    repository,
    user,
    req
  );

  const newImageId = await repository.image.add(newImageEntity, user);

  const createdImage = await repository.image.get({ id: newImageId }, user);

  if (createdImage == null) {
    throw new Error("An error has occurred during image creation");
  }
  return createdImage;
};

const deleteImage = async (
  _: any,
  args: MutationDeleteImageArgs,
  { repository, user }: Context
): Promise<DbImage> => {
  const imageId = args.where.id;
  const imageToDelete = await throwIfResolvesToNil(
    "No image with such id",
    repository.image.get
  )({ id: imageId }, user);
  const labelsToDelete = await repository.label.list({
    imageId,
    user,
  });
  await Promise.all(
    labelsToDelete.map((label) =>
      repository.label.delete({ id: label.id }, user)
    )
  );
  await repository.image.delete({ id: imageId }, user);
  await repository.upload.delete(imageToDelete.url);

  return imageToDelete;
};

const updateImage = async (
  _: any,
  args: MutationUpdateImageArgs,
  { repository, user }: Context
) => {
  const imageId = args.where.id;

  const now = new Date();

  const newImageEntity = {
    ...args.data,
    updatedAt: now.toISOString(),
  };

  await repository.image.update({ id: imageId }, newImageEntity, user);

  return await getImageById(imageId, repository, user);
};

const imagesAggregates = (parent: any) => {
  // Forward `parent` to chained resolvers if it exists
  return parent ?? {};
};

const totalCount = async (
  parent: any,
  _args: any,
  { repository, user }: Context
) => {
  // eslint-disable-next-line no-underscore-dangle
  const typename = parent?.__typename;
  if (typename === "Dataset") {
    return await repository.image.count({
      datasetId: parent.id,
      user,
    });
  }

  return await repository.image.count({ user });
};

export default {
  Query: {
    image,
    images,
    imagesAggregates,
  },

  Mutation: {
    createImage,
    updateImage,
    deleteImage,
  },

  Image: {
    labels: labelsResolver,
    thumbnail20Url: thumbnailResolver(20),
    thumbnail50Url: thumbnailResolver(50),
    thumbnail100Url: thumbnailResolver(100),
    thumbnail200Url: thumbnailResolver(200),
    thumbnail500Url: thumbnailResolver(500),
  },

  ImagesAggregates: { totalCount },

  Dataset: {
    imagesAggregates,
  },
};
