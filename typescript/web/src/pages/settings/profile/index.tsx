import { useCallback } from "react";
import { useSession } from "next-auth/react";
import { gql, useMutation, useQuery } from "@apollo/client";
import { Meta } from "../../../components/meta";
import { Layout } from "../../../components/layout";
import { ServiceWorkerManagerModal } from "../../../components/service-worker-manager";
import { AuthManager } from "../../../components/auth-manager";
import { WelcomeManager } from "../../../components/welcome-manager";
import { CookieBanner } from "../../../components/cookie-banner";
import { NavLogo } from "../../../components/logo/nav-logo";
import { Profile } from "../../../components/profile";

const updateUserQuery = gql`
  mutation updateUser($id: ID!, $data: UserUpdateInput!) {
    updateUser(where: { id: $id }, data: $data) {
      id
    }
  }
`;

const userQuery = gql`
  query getUserProfileInfo($id: ID!) {
    user(where: { id: $id }) {
      id
      createdAt
      name
      email
      image
    }
  }
`;

const ProfilePage = () => {
  const session = useSession({ required: false });
  const userInfoFromSession = session?.data?.user;

  const { data: userData } = useQuery(userQuery, {
    variables: { id: userInfoFromSession?.id },
    skip: userInfoFromSession?.id == null,
  });
  const user = userData?.user;
  const [updateUser] = useMutation(updateUserQuery, {
    refetchQueries: ["getUserProfileInfo"],
  });
  const changeUserName = useCallback(
    (name: string) => {
      updateUser({ variables: { id: user?.id, data: { name } } });
    },
    [updateUser, user]
  );
  if (user == null) {
    return null;
  }
  return (
    <>
      <ServiceWorkerManagerModal />
      <WelcomeManager />
      <AuthManager />
      <Meta title="LabelFlow | Profile" />
      <CookieBanner />
      <Layout breadcrumbs={[<NavLogo key={0} />]}>
        <Profile user={user} changeUserName={changeUserName} />
      </Layout>
    </>
  );
};

export default ProfilePage;