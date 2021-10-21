import { ApolloProvider, ApolloError } from "@apollo/client";
import React from "react";

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { forbiddenWorkspaceSlugs } from "@labelflow/common-resolvers";
import { WorkspaceCreationModal, Message } from "..";
import { client } from "../../../../connectors/apollo-client/schema-client";

jest.mock("use-query-params", () => ({
  useQueryParam: () => jest.fn(),
}));

describe("Message", () => {
  it("renders the future url if it is possible", () => {
    const { getByText } = render(
      <Message
        error={undefined}
        workspaceName="test"
        workspaceNameIsAlreadyTaken={false}
      />
    );
    expect(
      getByText(/Your URL will be: http:\/\/localhost\/test/)
    ).toBeDefined();
  });

  it("warns if the name is already taken", () => {
    const { getByText } = render(
      <Message
        error={undefined}
        workspaceName="test"
        workspaceNameIsAlreadyTaken
      />
    );
    expect(getByText(/The name "test" is already taken/)).toBeDefined();
  });

  it("warns if the name is a reserved name", () => {
    const { getByText } = render(
      <Message
        error={undefined}
        workspaceName={forbiddenWorkspaceSlugs[0]}
        workspaceNameIsAlreadyTaken={false}
      />
    );
    expect(getByText(/The name ".*?" is already taken/)).toBeDefined();
  });

  it("warns if the name contains invalid characters", () => {
    const { getByText } = render(
      <Message
        error={undefined}
        workspaceName="hello!"
        workspaceNameIsAlreadyTaken={false}
      />
    );
    expect(
      getByText('The name "hello!" contains invalid characters.')
    ).toBeDefined();
  });

  it("displays the error if given one", () => {
    const { getByText } = render(
      <Message
        error={new ApolloError({ errorMessage: "this is an error" })}
        workspaceName="test"
        workspaceNameIsAlreadyTaken={false}
      />
    );
    expect(getByText(/this is an error/)).toBeDefined();
  });

  it("displays an empty line if no workspace name is provided", () => {
    const { container } = render(
      <Message
        error={undefined}
        workspaceName={undefined}
        workspaceNameIsAlreadyTaken={false}
      />
    );

    expect(container.getElementsByTagName("br").length).toEqual(1);
  });
});

describe("WorkspaceCreationModal", () => {
  const Wrapper = ({ children }: React.PropsWithChildren<{}>) => (
    <ApolloProvider client={client}>{children}</ApolloProvider>
  );
  it("renders a disabled button if no name is specified", () => {
    const { getByRole } = render(
      <WorkspaceCreationModal
        isOpen
        onClose={console.log}
        initialWorkspaceName={undefined}
      />,
      { wrapper: Wrapper }
    );
    expect(getByRole("button", { name: "Create" })).toBeDisabled();
  });

  it("can create if the input is valid", () => {
    const { getByRole } = render(
      <WorkspaceCreationModal
        isOpen
        onClose={console.log}
        initialWorkspaceName={undefined}
      />,
      { wrapper: Wrapper }
    );

    const input = screen.getByLabelText(
      /workspace name input/i
    ) as HTMLInputElement;

    userEvent.type(input, "My new workspace");

    expect(getByRole("button", { name: "Create" })).not.toBeDisabled();
  });

  it("cannot create if the input contains invalid characters", () => {
    const { getByRole } = render(
      <WorkspaceCreationModal
        isOpen
        onClose={console.log}
        initialWorkspaceName={undefined}
      />,
      { wrapper: Wrapper }
    );

    const input = screen.getByLabelText(
      /workspace name input/i
    ) as HTMLInputElement;

    userEvent.type(input, "My new workspace!");

    expect(getByRole("button", { name: "Create" })).toBeDisabled();
  });

  it("cannot create if the input is a reserved name", () => {
    const { getByRole } = render(
      <WorkspaceCreationModal
        isOpen
        onClose={console.log}
        initialWorkspaceName={undefined}
      />,
      { wrapper: Wrapper }
    );

    const input = screen.getByLabelText(
      /workspace name input/i
    ) as HTMLInputElement;

    userEvent.type(input, "settings");

    expect(getByRole("button", { name: "Create" })).toBeDisabled();
  });

  it("cannot create if the name is already taken", () => {
    const { getByRole } = render(
      <WorkspaceCreationModal
        isOpen
        onClose={console.log}
        initialWorkspaceName={undefined}
      />,
      { wrapper: Wrapper }
    );

    const input = screen.getByLabelText(
      /workspace name input/i
    ) as HTMLInputElement;

    userEvent.type(input, "local");

    expect(getByRole("button", { name: "Create" })).toBeDisabled();
  });

  it("pre-fills the workspace name", () => {
    render(
      <WorkspaceCreationModal
        isOpen
        onClose={console.log}
        initialWorkspaceName="Pre-filled workspace name"
      />,
      { wrapper: Wrapper }
    );

    const input = screen.getByLabelText(
      /workspace name input/i
    ) as HTMLInputElement;

    expect(input.value).toEqual("Pre-filled workspace name");
  });
});
