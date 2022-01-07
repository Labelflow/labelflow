import {
  chakra,
  Box,
  Button,
  Flex,
  FlexProps,
  HStack,
  useDisclosure,
  VisuallyHidden,
  Spacer,
  IconButton,
  // useColorModeValue as mode,
} from "@chakra-ui/react";
import { RiGithubFill } from "react-icons/ri";
import * as React from "react";
import NextLink from "next/link";
import { Logo } from "../Logo";
import { NavLink } from "./NavLink";
import { NavMenu } from "./NavMenu";
import { Submenu } from "./Submenu";
import { ToggleButton } from "./ToggleButton";
import { links } from "./_data";

const GithubIcon = chakra(RiGithubFill);

const MobileNavContext = (props: FlexProps) => {
  const { isOpen, onToggle } = useDisclosure();
  return (
    <>
      <Flex align="center" className="nav-content__mobile" {...props}>
        <Box>
          <ToggleButton isOpen={isOpen} onClick={onToggle} />
        </Box>
        <NextLink href="/website">
          <Box as="a" rel="home" mx="auto" cursor="pointer">
            <Logo h="24px" iconColor="brand.400" />
          </Box>
        </NextLink>
        <Spacer />
        <IconButton
          aria-label="See code on github"
          href="https://github.com/labelflow/labelflow"
          as="a"
          target="blank"
          variant="link"
          icon={<GithubIcon fontSize="2xl" />}
        />
        <Box display={{ base: "none", sm: "block" }}>
          <NextLink href="/test/datasets">
            <Button as="a" colorScheme="brand" variant="outline" ml="3">
              Try it now
            </Button>
          </NextLink>
        </Box>
      </Flex>
      <NavMenu animate={isOpen ? "open" : "closed"}>
        {links.map((link, idx) =>
          link.children ? (
            <Submenu.Mobile key={idx} link={link} />
          ) : (
            <NavLink.Mobile
              key={idx}
              active={false}
              href={link.href as string}
              target={link.target}
            >
              {link.label}
            </NavLink.Mobile>
          )
        )}
        <NextLink href="/test/datasets">
          <Button colorScheme="brand" w="full" size="lg" mt="5">
            Try it now
          </Button>
        </NextLink>
      </NavMenu>
    </>
  );
};

const DesktopNavContent = (props: FlexProps) => {
  return (
    <Flex
      className="nav-content__desktop"
      align="center"
      justify="space-between"
      {...props}
    >
      <NextLink href="/website">
        <Box as="a" rel="home" cursor="pointer">
          <VisuallyHidden>LabelFlow</VisuallyHidden>
          <Logo h="6" iconColor="brand.500" />
        </Box>
      </NextLink>
      <HStack
        as="ul"
        id="nav__primary-menu"
        aria-label="Main Menu"
        listStyleType="none"
      >
        {links.map((link, idx) => (
          <Box as="li" key={idx} id={`nav__menuitem-${idx}`}>
            {link.children ? (
              <Submenu.Desktop link={link} />
            ) : (
              <NavLink.Desktop
                active={false}
                href={link.href as string}
                target={link.target}
              >
                {link.label}
              </NavLink.Desktop>
            )}
          </Box>
        ))}
      </HStack>
      <HStack
        spacing="8"
        //  minW="240px"
        justify="space-between"
      >
        {/* <Box
          as="a"
          href="#"
          color={mode("brand.600", "brand.300")}
          fontWeight="bold"
        >
          Sign In
        </Box> */}
        <IconButton
          aria-label="See code on github"
          href="https://github.com/labelflow/labelflow"
          as="a"
          target="blank"
          variant="link"
          mr="-2"
          icon={<GithubIcon fontSize="2xl" />}
        />
        <NextLink href="/test/datasets">
          <Button
            as="a"
            href="#"
            colorScheme="brand"
            fontWeight="bold"
            variant="outline"
          >
            Try it now
          </Button>
        </NextLink>
      </HStack>
    </Flex>
  );
};

export const NavContent = {
  Mobile: MobileNavContext,
  Desktop: DesktopNavContent,
};
