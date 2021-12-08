import {
  Button,
  ButtonGroup,
  FormControl,
  FormLabel,
  HStack,
  Input,
  InputGroup,
  InputLeftElement,
  Stack,
  chakra,
  Alert,
  AlertIcon,
  AlertTitle,
} from "@chakra-ui/react";
import { useState } from "react";
import { IoSearch } from "react-icons/io5";
import { RiAddFill } from "react-icons/ri";
import { useRouter } from "next/router";
import { InvitationResult } from "@labelflow/graphql-types";

import { NewMemberModal } from "./new-member-modal";
import { InviteMember } from "./types";

const SearchIcon = chakra(IoSearch);

export const TableActions = ({
  searchText,
  setSearchText,
  inviteMember = async () => InvitationResult.Sent,
}: {
  searchText: string;
  setSearchText: (text: string) => void;
  inviteMember?: InviteMember;
}) => {
  const [isNewMemberModalOpen, setIsNewMemberModalOpen] = useState(false);
  const router = useRouter();
  const workspaceSlug = router?.query.workspaceSlug as string;

  const tableActionsForOnlineWorkspace = (
    <>
      <NewMemberModal
        isOpen={isNewMemberModalOpen}
        onClose={() => setIsNewMemberModalOpen(false)}
        inviteMember={inviteMember}
      />
      <Stack
        spacing="4"
        direction={{ base: "column", md: "row" }}
        justify="space-between"
      >
        <HStack>
          <FormControl minW={{ md: "320px" }} id="search">
            <InputGroup size="sm">
              <FormLabel srOnly>Find a member</FormLabel>
              <InputLeftElement pointerEvents="none" color="gray.400">
                <SearchIcon />
              </InputLeftElement>
              <Input
                rounded="base"
                type="search"
                placeholder="Find a member"
                bgColor="#FFFFFF"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
              />
            </InputGroup>
          </FormControl>
        </HStack>
        <ButtonGroup size="sm" variant="outline">
          <Button
            backgroundColor="brand.500"
            color="#ffffff"
            iconSpacing="1"
            leftIcon={<RiAddFill fontSize="1.25em" />}
            onClick={() => setIsNewMemberModalOpen(true)}
          >
            New member
          </Button>
        </ButtonGroup>
      </Stack>
    </>
  );

  const tableActionsForLocalWorkspace = (
    <Alert status="info" variant="subtle">
      <AlertIcon />
      <AlertTitle>This workspace is private to you only.</AlertTitle>
      Its datasets are saved on your device only, not online. To collaborate and
      invite people, switch to an online workspace.
    </Alert>
  );
  return workspaceSlug === "local"
    ? tableActionsForLocalWorkspace
    : tableActionsForOnlineWorkspace;
};
