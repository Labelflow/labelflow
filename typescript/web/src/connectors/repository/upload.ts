import type { UploadTarget, UploadTargetHttp } from "@labelflow/graphql-types";
import { isInWindowScope } from "../../utils/detect-scope";

export const uploadsCacheName = "uploads";
export const uploadsRoute = "/api/worker/uploads";

declare let self: ServiceWorkerGlobalScope;

export const getUploadTargetHttp = async (
  key: string,
  origin: string
): Promise<UploadTargetHttp> => {
  return {
    __typename: "UploadTargetHttp",
    // Upload and download URL do not have to be the same. But in our implementation it is:
    uploadUrl: `${origin}${uploadsRoute}/${key}`,
    downloadUrl: `${origin}${uploadsRoute}/${key}`,
  };
};

/**
 * A way for the server to tell how it wants to accept file uploads
 * @returns a presigned URL for the client to upload files to, or `null` if the server wants to accept direct graphql uploads
 */
export const getUploadTarget = async (key: string): Promise<UploadTarget> => {
  if (isInWindowScope) {
    // We run in the window scope
    return { __typename: "UploadTargetDirect", direct: true };
  }

  // We run in the worker scope or nodejs
  return await getUploadTargetHttp(key, self.location.origin);
};

export const putInStorage = async (url: string, file: Blob) => {
  const response = new Response(file, {
    status: 200,
    statusText: "OK",
    headers: new Headers({
      "Content-Type": file.type ?? "application/octet-stream",
      "Content-Length": file.size.toString() ?? "0",
    }),
  });

  await (await caches.open(uploadsCacheName)).put(url, response);
};

export const getFromStorage = async (url: string): Promise<ArrayBuffer> => {
  const cacheResult = await (await caches.open(uploadsCacheName)).match(url);

  const fetchResult =
    cacheResult ??
    (await fetch(url, {
      method: "GET",
      mode: "cors",
      headers: new Headers({
        // No need to add specific headers here, they are already added by the fetch call
        Accept: "image/tiff,image/jpeg,image/png,image/*,*/*;q=0.8",
        "Sec-Fetch-Dest": "image",
      }),
      credentials: "omit",
    }));

  if (fetchResult.status !== 200) {
    throw new Error(
      `From client, could not fetch image at url ${url} properly, code ${fetchResult.status}`
    );
  }
  return await fetchResult.arrayBuffer();
};

export const deleteFromStorage = async (url: string): Promise<void> => {
  const cache = await caches.open(uploadsCacheName);
  if (await cache.match(url)) {
    await cache.delete(url);
  }
};
