import {
  Box,
  Heading,
  Link,
  Text,
  useColorModeValue as mode,
} from "@chakra-ui/react";
import * as React from "react";

import { NavContent } from "../components/website/Navbar/NavContent";

import { Footer } from "../components/website/Footer/Footer";

import { Meta } from "../components/meta";

export default function Pricing() {
  return (
    <Box minH="640px">
      <Meta />

      <Box
        as="header"
        bg={mode("white", "gray.800")}
        position="relative"
        zIndex="10"
      >
        <Box
          as="nav"
          aria-label="Main navigation"
          maxW="7xl"
          mx="auto"
          px={{ base: "6", md: "8" }}
        >
          <NavContent.Mobile display={{ base: "flex", lg: "none" }} />
          <NavContent.Desktop display={{ base: "none", lg: "flex" }} />
        </Box>

        <Box
          as="section"
          //   bg={mode("gray.50", "gray.800")}
          py={{ base: "10", sm: "24" }}
        >
          <Box
            maxW={{ base: "xl", md: "3xl" }}
            mx="auto"
            px={{ base: "6", md: "8" }}
            className="markdown-body"
            boxSizing="border-box"
          >
            <Heading
              align="center"
              fontWeight="extrabold"
              maxW="lg"
              mx="auto"
              mb="20"
            >
              About us
            </Heading>
            <Text textAlign="justify">
              Over the last 5 years we have built{" "}
              <Link
                href="https://www.sterblue.com"
                isExternal
                color="brand.600"
              >
                Sterblue
              </Link>
              , an AI-powered cloud platform to support energy companies
              managing their critical assets: wind turbines, transmission &
              distribution grids, cooling towers. Machine learning was sweating
              all along our product.
              <br />
              <br />
              Millions of images have been through the platform. We have used AI
              to detect discrepancies (corrosion, cracks, vegetation, etc.), to
              locate asset in space, to navigate automatically using drones, and
              many other applications.
              <br />
              <br />
              5 years ago, showing bounding boxes on an image on a Linkedin or
              blog post was the big marketing trend. We took the hard way, the
              one that targetted AI at scale.
              <br />
              <br />
              We labeled millions of images dealing with multiple labelling
              companies, we dealt with tens of different taxonomies for the same
              physical problem, we faced labelling quality challenges, we build
              AI training pipeline using tens of open source frameworks and
              ecosystem partners, we integrated real users in the loop and ...
              we showed some results.
              <br />
              <br />
              The type of results that says to a customer “for your use case A,
              you save 23% of your time increasing false negative rate by 14%”
              <br />
              <br />
              With Labelflow, we now want to support every AI-applied company in
              the world building the next big thing. We start small with our
              labelling tool, with a strong focus on user experience (here).
              That’s just the beginning.
              <br />
              <br />
              We are a team of enthusiastic developers and data scientists. We
              will unveil the team soon, stay tuned.
            </Text>
          </Box>
        </Box>
      </Box>
      <Footer />
    </Box>

    // <Box minH="640px">
    //   <Meta title="Labelflow - Pricing" />

    //   <Box
    //     as="header"
    //     bg={mode("white", "gray.800")}
    //     position="relative"
    //     zIndex="10"
    //   >
    //     <Box
    //       as="nav"
    //       aria-label="Main navigation"
    //       maxW="7xl"
    //       mx="auto"
    //       px={{ base: "6", md: "8" }}
    //     >
    //       <NavContent.Mobile display={{ base: "flex", lg: "none" }} />
    //       <NavContent.Desktop display={{ base: "none", lg: "flex" }} />
    //     </Box>
    //   </Box>

    //   <PricingTable />
    //   <Footer />
    // </Box>
  );
}