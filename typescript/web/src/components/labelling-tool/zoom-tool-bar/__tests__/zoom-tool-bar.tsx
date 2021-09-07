import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ZoomToolbar } from "../zoom-tool-bar";
import { useLabellingStore } from "../../../../connectors/labeling-state";

beforeEach(() => {
  jest.clearAllMocks();
});

it("should display zoom in and out buttons", () => {
  render(<ZoomToolbar />);

  expect(screen.getByRole("button", { name: "Zoom out" })).toBeDefined();
  expect(screen.getByRole("button", { name: "Zoom in" })).toBeDefined();
});

it("should disable zoom out by default", () => {
  render(<ZoomToolbar />);

  expect(screen.getByRole("button", { name: "Zoom out" })).toBeDisabled();
});

it("should disable zoom in if image is at minimal resolution", async () => {
  render(<ZoomToolbar />);

  await waitFor(() => {
    useLabellingStore.setState({ canZoomIn: false });
  });

  expect(screen.getByRole("button", { name: "Zoom in" })).toBeDisabled();
});

it("should disable zoom out if image is at maximum resolution", async () => {
  useLabellingStore.setState({ canZoomOut: true });
  render(<ZoomToolbar />);

  await waitFor(() => {
    useLabellingStore.setState({ canZoomOut: false });
  });

  expect(screen.getByRole("button", { name: "Zoom out" })).toBeDisabled();
});

useLabellingStore.setState({
  zoomByDelta: jest.fn(),
  setView: jest.fn(),
  zoomFactor: 0.6,
});

it("should zoom out by a zoom factor", async () => {
  useLabellingStore.setState({ canZoomOut: true });
  render(<ZoomToolbar />);

  userEvent.click(screen.getByRole("button", { name: "Zoom out" }));

  const state = useLabellingStore.getState();
  expect(state.zoomByDelta).toHaveBeenCalledWith(-state.zoomFactor);
});

it("should zoom in by a zoom factor", async () => {
  useLabellingStore.setState({ canZoomIn: true });
  render(<ZoomToolbar />);

  userEvent.click(screen.getByRole("button", { name: "Zoom in" }));

  const state = useLabellingStore.getState();
  expect(state.zoomByDelta).toHaveBeenCalledWith(state.zoomFactor);
});
