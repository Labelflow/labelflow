import { HStack, Text } from "@chakra-ui/react";
import { sleep } from "@labelflow/utils";
import { ComponentMeta, ComponentStory } from "@storybook/react";
import { userEvent, within } from "@storybook/testing-library";
import { chakraDecorator, storybookTitle } from "../../utils/stories";
import { Tooltip } from "./tooltip";

export default {
  title: storybookTitle(Tooltip),
  component: Tooltip,
  decorators: [chakraDecorator],
} as ComponentMeta<typeof Tooltip>;

const Template: ComponentStory<typeof Tooltip> = (args) => (
  <HStack align="flex-start">
    <Tooltip data-testid="tooltip" shouldWrapChildren {...args}>
      <Text>Hover me</Text>
    </Tooltip>
  </HStack>
);

export const Normal = Template.bind({});
Normal.args = { label: "Normal" };

export const Hovered = Template.bind({});
Hovered.args = { label: "Hovered" };
Hovered.play = async ({ canvasElement }) => {
  const { getByTestId } = within(canvasElement);
  const card = getByTestId("tooltip");
  userEvent.hover(card);
  // Wait for the tooltip to show before taking a snapshot
  await sleep(500);
};

export const Opened = Template.bind({});
Opened.args = { label: "Opened", isOpen: true };
