import { ComponentMeta, ComponentStory } from "@storybook/react";
import {
  chakraDecorator,
  modalDecorator,
  storybookTitle,
} from "../../../utils/stories";
import { DeleteModal as DeleteModalComponent } from "./delete-modal";

export default {
  title: storybookTitle("Core", DeleteModalComponent),
  component: DeleteModalComponent,
  decorators: [chakraDecorator, modalDecorator],
} as ComponentMeta<typeof DeleteModalComponent>;

const Template: ComponentStory<typeof DeleteModalComponent> = (args) => (
  <DeleteModalComponent {...args} />
);

export const Default = Template.bind({});
Default.args = {
  isOpen: true,
  header: "Delete FooBar",
  body: "Are you sure that you want to delete this resource?",
};

export const Deleting = Template.bind({});
Deleting.args = {
  ...Default.args,
  deleting: true,
};
Deleting.parameters = { chromatic: { disableSnapshot: true } };