/* eslint-disable import/first */
import { ApolloProvider } from "@apollo/client";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom/extend-expect";
import { client } from "../../../../connectors/apollo-client/schema-client";
import { mockNextRouter } from "../../../../utils/router-mocks";
import { useLabellingStore } from "../../../../connectors/labeling-state";

mockNextRouter();

import { DrawingToolbar } from "..";

test("should display tooltip", async () => {
  useLabellingStore.setState({ isImageLoading: false });

  render(<DrawingToolbar />, {
    wrapper: ({ children }) => (
      <ApolloProvider client={client}>{children}</ApolloProvider>
    ),
  });

  const selectionToolButton = await screen.getByLabelText(/Selection tool/i);

  userEvent.hover(selectionToolButton as HTMLElement);

  await waitFor(() => expect(screen.getByText(/\[v\]/i)).toBeInTheDocument());

  expect(screen.queryByText(/Selection tool/i)).toBeInTheDocument();
});
