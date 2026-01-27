import { PropsWithChildren, useMemo } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ConfigProvider, theme as antdTheme } from "antd";
import { ThemeProvider, useTheme } from "../components/theme-provider";

function createClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        retry: false,
      },
    },
  });
}

function AntdConfigProvider({ children }: PropsWithChildren) {
  const { theme } = useTheme();

  return (
    <ConfigProvider
      theme={{
        algorithm: theme === "dark" ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
        token: {
          colorPrimary: "#00beb4",
          borderRadius: 8,
          fontFamily: "Inter, system-ui, -apple-system, sans-serif",
        },
      }}
    >
      {children}
    </ConfigProvider>
  );
}

export default function AppProviders({ children }: PropsWithChildren) {
  const client = useMemo(() => createClient(), []);

  return (
    <QueryClientProvider client={client}>
      <ThemeProvider>
        <AntdConfigProvider>{children}</AntdConfigProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
