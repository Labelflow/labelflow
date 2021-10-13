const withPWA = require("next-pwa");
// const { BundleAnalyzerPlugin } = require("webpack-bundle-analyzer");
const path = require("path");

// This file sets a custom webpack configuration to use your Next.js app
// with Sentry.
// https://nextjs.org/docs/api-reference/next.config.js/introduction
// https://docs.sentry.io/platforms/javascript/guides/nextjs/

// const { withSentryConfig } = require("@sentry/nextjs");

// const SentryWebpackPluginOptions = {
//   // Additional config options for the Sentry Webpack plugin. Keep in mind that
//   // the following options are set automatically, and overriding them is not
//   // recommended:
//   //   release, url, org, project, authToken, configFile, stripPrefix,
//   //   urlPrefix, include, ignore

//   silent: true, // Suppresses all logs
//   // For all available options, see:
//   // https://github.com/getsentry/sentry-webpack-plugin#options.
// };

//withSentryConfig(
// withPWA(
module.exports = withPWA({
  reactStrictMode: true,
  sentry: {
    disableServerWebpackPlugin:
      process.env.SENTRY_AUTH_TOKEN != null ? false : true,
    disableClientWebpackPlugin:
      process.env.SENTRY_AUTH_TOKEN != null ? false : true,
  },
  images: {
    deviceSizes: [
      320, 480, 640, 750, 828, 960, 1080, 1200, 1440, 1920, 2048, 2560, 3840,
    ],
  },
  experimental: {
    esmExternals: false,
    swcLoader: true,
    swcMinify: false,
  },
  webpack5: true,
  typescript: {
    ignoreDevErrors: true,
    ignoreBuildErrors: true,
  },
  webpack: (
    config,
    { defaultLoaders, dev, isServer, config: nextConfig, ...others }
  ) => {
    // Note: we provide webpack above so you should not `require` it
    // Perform customizations to webpack config

    // Add graphql import
    // See https://www.npmjs.com/package/graphql-tag#webpack-loading-and-preprocessing
    config.module.rules = [
      ...config.module.rules,
      {
        test: /\.(graphql|gql)$/,
        use: "graphql-tag/loader",
        exclude: /node_modules/,
      },
    ];

    // Transpile other packages of the monorepo
    // E.g.: `@labelflow/react-openlayers-fiber`
    // See https://github.com/vercel/next.js/blob/0af3b526408bae26d6b3f8cab75c4229998bf7cb/test/integration/typescript-workspaces-paths/packages/www/next.config.js
    // The root folder of the monorepo:
    const resolvedBaseUrl = path.resolve(config.context, "../../");
    // We add a rule to transpile files in this folder, except files in `node_modules`
    config.module.rules = [
      ...config.module.rules,
      {
        test: /\.(tsx|ts|js|mjs|jsx)$/,
        use: defaultLoaders.babel,
        include: [resolvedBaseUrl],
        exclude: (excludePath) => {
          // To allow to resolve files inside `node_modules`, we could add a condition like this:
          //     return /node_modules/.test(excludePath) && ! /\/ol/.test(excludePath)
          // This is not needed for now though:
          return /node_modules/.test(excludePath);
        },
      },
    ];

    //   // // Transpile node-modules packages that provides no nodejs compatible files
    //   // // E.g.: `ol`
    //   // // See https://github.com/vercel/next.js/issues/9890
    //   // // See https://github.com/openlayers/openlayers/issues/10470
    //   // // See https://github.com/vercel/next.js/blob/bd589349d2a90c41e7fc9549ea2438febfc9a510/packages/next/build/webpack-config.ts#L637
    //   // // To do this we modify the `externals` option of webpack:
    //   // config.externals = config.externals.map((external) => {
    //   //   if (!(typeof external === "function")) {
    //   //     // `externals` options that are hardcoded strings and arrays are used directly
    //   //     return external;
    //   //   } else {
    //   //     // `externals` options that are functions are overridden, to force externalize of the packages we want

    //   //     if (isWebpack5) {
    //   //       // Return a webpack5-like `externals` option function
    //   //       return (
    //   //         { context, request, contextInfo, getResolve },
    //   //         callback
    //   //       ) => {
    //   //         if (/^ol/.test(request)) {
    //   //           // Make an exception for `ol`, never externalize this import, it must be transpiled and bundled
    //   //           return callback?.();
    //   //         } else {
    //   //           // Use the standard NextJS `externals` function
    //   //           return external(
    //   //             { context, request, contextInfo, getResolve },
    //   //             callback
    //   //           );
    //   //         }
    //   //       };
    //   //     } else {
    //   //       // Return a webpack4-like `externals` option function
    //   //       return (context, request, callback) => {
    //   //         if (/^ol/.test(request)) {
    //   //           // Make an exception for `ol`, never externalize this import, it must be transpiled and bundled
    //   //           return callback?.();
    //   //         } else {
    //   //           // Use the standard NextJS `externals` function
    //   //           return external(context, request, callback);
    //   //         }
    //   //       };
    //   //     }
    //   //   }
    //   // });

    // Allow to transpile node modules that depends on node built - ins into browser.
    //   E.g.: `apollo-server-core`
    // See https://github.com/webpack-contrib/css-loader/issues/447
    // See https://github.com/vercel/next.js/issues/7755
    if (!isServer) {

      // See https://www.npmjs.com/package/node-polyfill-webpack-plugin
      const NodePolyfillPlugin = require("node-polyfill-webpack-plugin");

      // See https://github.com/webpack-contrib/css-loader/issues/447#issuecomment-761853289
      // See https://github.com/vercel/next.js/issues/7755#issuecomment-812805708
      config.resolve = {
        ...(config.resolve ?? {}),
        fallback: {
          ...(config.resolve?.fallback ?? {}),
          module: false,
          dgram: false,
          dns: false,
          fs: false,
          http2: false,
          net: false,
          tls: false,
          child_process: false,
        },
      };
      config.plugins = [
        ...(config?.plugins ?? []),
        new NodePolyfillPlugin({
          excludeAliases: ["console"],
        }),
      ];

    }

    // Add webpack bundle analyzer with custom config to expose the reports publicly
    // See https://github.com/vercel/next.js/blob/canary/packages/next-bundle-analyzer/index.js
    if (!dev) {
      config.plugins.push(
        new BundleAnalyzerPlugin({
          analyzerMode: "static",
          openAnalyzer: false,
          reportFilename: isServer
            ? "../../static/bundle-analyzer/server.html"
            : "./static/bundle-analyzer/client.html",
        })
      );
    }

    // // Enable top level await for apollo-server-micro in `typescript/web/src/pages/api/graphql.ts`
    // // See https://stackoverflow.com/questions/68339243/how-can-i-use-top-level-await-in-typescript-next-js
    // config.experiments = {
    //   ...config.experiments,
    //   topLevelAwait: true,
    // };

    // Important: return the modified config
    return config;
  },
  // Make sure entries are not getting disposed.
  // See https://github.com/vercel/next.js/blob/0af3b526408bae26d6b3f8cab75c4229998bf7cb/test/integration/typescript-workspaces-paths/packages/www/next.config.js
  onDemandEntries: {
    maxInactiveAge: 1000 * 60 * 60,
  },
  // Put the Service Worker code in the `public` folder to avoid having to serve it separately
  // See https://github.com/shadowwalker/next-pwa#usage-without-custom-server-nextjs-9
  pwa: {
    dest: "public",
    swSrc: "./src/worker/index.ts",
    compileSrc: true,
    // Register false, since we register manually in `_app.tsx`, and ask the user when to upgrade
    register: false,
    // Cache on frontend nav, it pre-fetches stuff more eagerly
    // See https://github.com/shadowwalker/next-pwa#available-options
    cacheOnFrontEndNav: true,
    // Add plugins to the webpack config of the service worker bundler
    // See https://developers.google.com/web/tools/workbox/reference-docs/latest/module-workbox-webpack-plugin.InjectManifest
    webpackCompilationPlugins: [],
    // exclude: ["/api/worker/"]
  },
});
//   ,
//   // ),
//   SentryWebpackPluginOptions
// );
