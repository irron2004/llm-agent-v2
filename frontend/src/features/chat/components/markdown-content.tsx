import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { CopyOutlined, CheckOutlined } from "@ant-design/icons";
import { useState } from "react";

type MarkdownContentProps = {
  content: string;
};

export function MarkdownContent({ content }: MarkdownContentProps) {
  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeRaw]}
        components={{
          code({ node, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || "");
            const isInline = !match && !className;

            if (isInline) {
              return (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            }

            return (
              <CodeBlock language={match?.[1] || "text"}>
                {String(children).replace(/\n$/, "")}
              </CodeBlock>
            );
          },
          // Custom table wrapper for horizontal scroll
          table({ children }) {
            return (
              <div style={{ overflowX: "auto" }}>
                <table>{children}</table>
              </div>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

type CodeBlockProps = {
  language: string;
  children: string;
};

function CodeBlock({ language, children }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(children);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy code:", err);
    }
  };

  return (
    <div style={{ position: "relative" }}>
      <div
        style={{
          position: "absolute",
          top: 8,
          right: 8,
          display: "flex",
          alignItems: "center",
          gap: 8,
          zIndex: 1,
        }}
      >
        <span
          style={{
            fontSize: 11,
            color: "var(--color-text-disabled)",
            textTransform: "uppercase",
          }}
        >
          {language}
        </span>
        <button
          onClick={handleCopy}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 24,
            height: 24,
            border: "none",
            borderRadius: 4,
            background: "rgba(255, 255, 255, 0.1)",
            color: "var(--color-code-text)",
            cursor: "pointer",
          }}
          title="Copy code"
        >
          {copied ? <CheckOutlined /> : <CopyOutlined />}
        </button>
      </div>
      <SyntaxHighlighter
        style={oneDark}
        language={language}
        PreTag="div"
        customStyle={{
          margin: 0,
          borderRadius: "var(--radius-sm)",
          fontSize: "var(--font-size-xs)",
        }}
      >
        {children}
      </SyntaxHighlighter>
    </div>
  );
}
