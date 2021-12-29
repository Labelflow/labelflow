import NextAuth, { Profile } from "next-auth";

import EmailProvider from "next-auth/providers/email";
import GoogleProvider from "next-auth/providers/google";
import GitHubProvider from "next-auth/providers/github";
import { PrismaAdapter } from "@next-auth/prisma-adapter";
import { PrismaClient as PrismaClientClass } from "@prisma/client";
import { OAuthConfig } from "next-auth/providers";
import { captureException } from "@sentry/nextjs";
import { createPrismaClient } from "@labelflow/db/src/prisma-client";

import { sendVerificationRequestFromPrisma } from "../../../utils/email/send-verification-request";

// interface NextAuthUserWithStringId extends NextAuthUser {
//   id: string;
// }

// Try to use the prisma singleton defined in typescript/db/src/prisma-client.ts
declare module globalThis {
  let prismaInstance: PrismaClientClass;
  let prismaInstanceIsConnected: boolean;
}
if (!globalThis.prismaInstance) {
  console.log("[Prisma Client] Initializing prismaInstance from next auth");
  globalThis.prismaInstance = createPrismaClient();
}
globalThis.prismaInstanceIsConnected = true;

export default NextAuth({
  providers: [
    EmailProvider({
      server: process.env.EMAIL_SERVER,
      from: process.env.EMAIL_FROM,
      sendVerificationRequest: sendVerificationRequestFromPrisma(
        globalThis.prismaInstance
      ),
    }),
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID as string,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET as string,
    }) as OAuthConfig<Profile>,
    GitHubProvider({
      clientId: process.env.GITHUB_CLIENT_ID,
      clientSecret: process.env.GITHUB_CLIENT_SECRET,
    }),
  ],
  adapter: PrismaAdapter(globalThis.prismaInstance),
  // Uncomment to implement your custom pages
  pages: {
    signIn: "/auth/signin",
    // signOut: "/auth/signout",
    // error: "/auth/error", // Error code passed in query string as ?error=
    // verifyRequest: "/auth/verify-request", // (used for check email message)
    // newUser: "/auth/new-user", // New users will be directed here on first sign in (leave the property out if not of interest)
  },
  // debug: true,
  callbacks: {
    session: ({ session, user }) =>
      // : {
      //   session: Session;
      //   user: NextAuthUserWithStringId;
      // }
      {
        if (session?.user && user?.id) {
          // eslint-disable-next-line no-param-reassign
          session.user.id = user?.id;
        }
        return session;
      },
  },
  logger: {
    error(code, metadata) {
      console.error(code, metadata);
      captureException(metadata);
    },
    warn(code) {
      console.warn(code);
    },
    debug(code, metadata) {
      console.debug(code, metadata);
    },
  },
});
