import {
  ButtonGroup,
  ButtonGroupProps,
  //  IconButton
} from "@chakra-ui/react";
import * as React from "react";
// import { FaGithub, FaLinkedin, FaTwitter } from "react-icons/fa";

export const SocialMediaLinks = (props: ButtonGroupProps) => (
  <ButtonGroup variant="ghost" color="gray.600" {...props}>
    {/* <IconButton
      as="a"
      href="https://www.linkedin.com/company/sterblue/"
      aria-label="LinkedIn"
      icon={<FaLinkedin fontSize="20px" />}
    />
    <IconButton
      as="a"
      href="https://github.com/sterblue"
      aria-label="GitHub"
      icon={<FaGithub fontSize="20px" />}
    />
    <IconButton
      as="a"
      href="https://twitter.com/Sterblue"
      aria-label="Twitter"
      icon={<FaTwitter fontSize="20px" />}
    /> */}
  </ButtonGroup>
);
