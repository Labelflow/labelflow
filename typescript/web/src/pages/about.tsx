import {
  Box,
  Heading,
  Link,
  Center,
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
              , <b>an AI-powered cloud platform to support energy companies </b>
              managing their critical assets: wind turbines, transmission &
              distribution grids, cooling towers. Machine learning was a huge
              part of our product.
              <br />
              <br />
              Millions of images have been processed on the platform. We have
              used AI to detect discrepancies (corrosion, cracks, vegetation,
              etc.), to locate assets in space, to generate automatic flight
              paths for drones, and many other applications.
              <br />
              <br />
              <Center>
                <img
                  src="static/img/about-screenshot1.jpg"
                  alt="Sterblue Platform"
                />
              </Center>
              <br />
              <br />
              5 years ago, showing bounding boxes on an image on a Linkedin or
              blog post was the big marketing trend. We chose to do it the hard
              way, by scaling up with AI.
              <br />
              <br />
              We labeled millions of images worked with multiple labeling
              companies. We also dealt with tens of different taxonomies for the
              same physical problem, we faced labeling quality challenges, we
              built AI training pipeline using tens of open source frameworks
              and ecosystem partners, we integrated real users in the loop and
              ... we showed some results.
              <br />
              <br />
              The type of results that tell to a customer “for your use case A,
              you save 23% of your time whilst reducing your false negative rate
              by 14%”
              <br />
              <br />
              With{" "}
              <b>
                <Link color="brand.500" href="https://labelflow.ai/website">
                  Labelflow
                </Link>
                , we now want to support every AI-applied company in the world
                to build the next big thing
              </b>
              . We start small with our labeling tool, with a strong focus on
              user experience. That’s just the beginning.
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
