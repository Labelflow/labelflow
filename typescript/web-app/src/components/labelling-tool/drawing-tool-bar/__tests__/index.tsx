import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom/extend-expect";

import { DrawingToolbar } from "..";

test("should display tooltip", async () => {
  render(<DrawingToolbar />);

  const selectionToolButton = await screen.getByLabelText(/Select tool/i);

  userEvent.hover(selectionToolButton as HTMLElement);

  await waitFor(() => expect(screen.getByText(/\[v\]/i)).toBeInTheDocument());

  expect(screen.queryByText(/Selection tool/i)).toBeInTheDocument();
});
