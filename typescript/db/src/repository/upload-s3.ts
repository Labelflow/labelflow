import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import "isomorphic-fetch";
import memoizeOne from "memoize-one";
import { Repository } from "../../../common-resolvers/src";
import { UploadTargetHttp } from "../../../graphql-types/src/graphql-types.generated";

const bucket = "labelflow";
const region = process.env?.LABELFLOW_AWS_REGION!;

const getClient = memoizeOne(
  () =>
    new S3Client({
      ...{
        region,
        credentials: {
          accessKeyId: process.env?.LABELFLOW_AWS_ACCESS_KEY_ID!,
          secretAccessKey: process.env?.LABELFLOW_AWS_SECRET_ACCESS_KEY!,
        },
      },
      // Uncomment bellow to try making minio work with s3 sdk
      // See https://docs.min.io/docs/how-to-use-aws-sdk-for-javascript-with-minio-server.html
      // // If we set an given endpoint, it means that we test with minio
      ...(process.env?.LABELFLOW_AWS_ENDPOINT
        ? {
            // endpoint: process.env?.LABELFLOW_AWS_ENDPOINT,
            endpoint: {
              protocol: "http",
              hostname: "localhost:9000",
              path: "/",
            },
            forcePathStyle: true, // required to make minio work
          }
        : {}),
    })
);

export const downloadsRoute = "/api/downloads";

export const getUploadTargetHttp = async (
  key: string,
  origin: string
): Promise<UploadTargetHttp> => {
  if (!key)
    return {
      __typename: "UploadTargetHttp",
      uploadUrl: "",
      downloadUrl: `${origin}${downloadsRoute}/`,
    };
  const s3Client = getClient();
  const command = new PutObjectCommand({
    Bucket: bucket,
    Key: key,
  });
  const signedUrl = await getSignedUrl(s3Client, command, {
    expiresIn: 3600,
  });
  return {
    __typename: "UploadTargetHttp",
    uploadUrl: signedUrl,
    downloadUrl: `${origin}${downloadsRoute}/${key}`,
  };
};

export const getFromStorage: Repository["upload"]["get"] = async (url, req) => {
  const headers = new Headers();
  headers.set("Accept", "image/tiff,image/jpeg,image/png,image/*,*/*;q=0.8");
  headers.set("Sec-Fetch-Dest", "image");
  if ((req?.headers as any)?.cookie) {
    headers.set("Cookie", (req?.headers as any)?.cookie);
  }

  const fetchResult = await fetch(url, {
    method: "GET",
    headers,
    credentials: "include",
  });

  if (fetchResult.status !== 200) {
    throw new Error(
      `Getting from S3 storage, could not fetch image at url ${url} properly, code ${fetchResult.status}`
    );
  }
  return await fetchResult.arrayBuffer();
};

export const deleteFromStorage: Repository["upload"]["delete"] = async (
  url
) => {
  console.warn(`
      deleteFromStorage called on AWS S3 uploader with url ${url}.
      Won't make anything
      `);
};

export const putInStorage: Repository["upload"]["put"] = async (url, blob) => {
  const fetchResult = await fetch(url, {
    method: "PUT",
    body: await blob.arrayBuffer(),
  });

  if (fetchResult.status !== 200) {
    throw new Error(
      `Putting to S3 storage, could not put at url ${url} properly, code ${fetchResult.status}`
    );
  }
};
