import gql from "graphql-tag";
import { useQuery } from "@apollo/client";
import NextLink from "next/link";
import { SimpleGrid, Breadcrumb, BreadcrumbItem, Text } from "@chakra-ui/react";
import { RiArrowRightSLine } from "react-icons/ri";

import { Meta } from "../../components/meta";
import { Layout } from "../../components/layout";
import { NewProjectCard, ProjectCard } from "../../components/projects";
import type { Project as ProjectType } from "../../graphql-types.generated";

const getProjectsQuery = gql`
  query getProjects {
    projects {
      id
      name
      images {
        url
      }
    }
  }
`;

const ProjectPage = () => {
  const { data: projectsResult } =
    useQuery<{ projects: Pick<ProjectType, "id" | "name" | "images">[] }>(
      getProjectsQuery
    );

  return (
    <>
      <Meta title="Labelflow | Projects" />
      <Layout
        topBarLeftContent={
          <Breadcrumb
            spacing="8px"
            separator={<RiArrowRightSLine color="gray.500" />}
          >
            <BreadcrumbItem>
              <Text>Projects</Text>
            </BreadcrumbItem>
          </Breadcrumb>
        }
      >
        <SimpleGrid
          gap={12}
          padding={12}
          minChildWidth="24rem"
          justifyItems="center"
        >
          <NewProjectCard />
          {projectsResult?.projects?.map(({ id, name }) => (
            <NextLink href={`/projects/${id}`} key={id}>
              {/* eslint-disable-next-line jsx-a11y/anchor-is-valid */}
              <a style={{ width: "100%" }}>
                <ProjectCard
                  projectName={name}
                  imageUrl="https://images.unsplash.com/photo-1579513141590-c597876aefbc?auto=format&fit=crop&w=882&q=80"
                  imagesCount={0}
                  labelClassesCount={0}
                  labelsCount={0}
                  editProject={() => {}}
                  deleteProject={() => {}}
                />
              </a>
            </NextLink>
          ))}
        </SimpleGrid>
      </Layout>
    </>
  );
};

export default ProjectPage;
