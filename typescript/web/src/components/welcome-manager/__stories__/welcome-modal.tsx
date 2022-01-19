import { chakraDecorator } from "../../../utils/chakra-decorator";
import { queryParamsDecorator } from "../../../utils/query-params-decorator";
import { apolloDecorator } from "../../../utils/apollo-decorator";
import { cookieDecorator } from "../../../utils/cookie-decorator";
import { WelcomeManager } from "..";

export default {
  title: "web/app lifecycle/welcome modal",
  decorators: [
    chakraDecorator,
    apolloDecorator,
    queryParamsDecorator,
    cookieDecorator,
  ],
};

export const Closed = () => {
  return <WelcomeManager />;
};

Closed.parameters = {
  nextRouter: {
    path: "/local/datasets",
    asPath: "/local/datasets",
    query: {},
  },
  cookie: {
    hasUserTriedApp: "true",
  },
};

export const BrowserError = () => {
  return (
    <WelcomeManager
      initialBrowserError={new Error("Wow")}
      initialBrowserWarning
    />
  );
};

BrowserError.parameters = {
  nextRouter: {
    path: "/local/datasets?modal-welcome=open",
    asPath: "/local/datasets?modal-welcome=open",
    query: {
      "modal-welcome": "open",
    },
  },
  cookie: {
    hasUserTriedApp: "false",
  },
};

export const BrowserWarning = () => {
  return <WelcomeManager initialBrowserWarning />;
};

BrowserWarning.parameters = {
  nextRouter: {
    path: "/local/datasets?modal-welcome=open",
    asPath: "/local/datasets?modal-welcome=open",
    query: {
      "modal-welcome": "open",
    },
  },
  cookie: {
    hasUserTriedApp: "false",
    tryDespiteBrowserWarning: "false",
  },
};

export const Loading = () => {
  return <WelcomeManager initialIsLoadingWorkerAndDemo />;
};

Loading.parameters = {
  nextRouter: {
    path: "/local/datasets?modal-welcome=open",
    asPath: "/local/datasets?modal-welcome=open",
    query: {
      "modal-welcome": "open",
    },
  },
  cookie: {
    hasUserTriedApp: "false",
    tryDespiteBrowserWarning: "false",
  },
};

/**
 * The Welcome Modal displays a countdown.
 * This is non-deterministic, so we can't do snapshot testing here.
 * The next story is the same welcome modal, but without a countdown.
 * So, it's deterministic and can be snapshot tested.
 */
export const Welcome = () => {
  return <WelcomeManager />;
};

Welcome.parameters = {
  chromatic: { disableSnapshot: true },
};

export const WelcomeWithoutCountDown = () => {
  return <WelcomeManager autoStartCountDown={false} />;
};

WelcomeWithoutCountDown.parameters = {
  nextRouter: {
    path: "/local/datasets?modal-welcome=open",
    asPath: "/local/datasets?modal-welcome=open",
    query: {
      "modal-welcome": "open",
    },
  },
  cookie: {
    hasUserTriedApp: "false",
    tryDespiteBrowserWarning: "false",
  },
};
