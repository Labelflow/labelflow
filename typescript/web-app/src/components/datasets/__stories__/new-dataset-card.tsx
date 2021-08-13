import { withNextRouter } from "storybook-addon-next-router";

import { Box } from "@chakra-ui/react";
import { chakraDecorator } from "../../../utils/chakra-decorator";
import { apolloDecorator } from "../../../utils/apollo-decorator";
import { queryParamsDecorator } from "../../../utils/query-params-decorator";

import { NewDatasetCard } from "..";

export default {
  title: "web-app/New Dataset Card",
  decorators: [
    chakraDecorator,
    apolloDecorator,
    queryParamsDecorator,
    withNextRouter,
  ],
};

export const Default = () => {
  return (
    <Box background="gray.100" padding={4} w="sm">
      <NewDatasetCard addDataset={() => {}} />
    </Box>
  );
};