import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Q-CHAT-10 Screening",
  description:
    "Toddler autism traits screening built on the Q-CHAT-10 instrument, with model comparison.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=JetBrains+Mono:wght@400;500&family=Public+Sans:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <header className="border-b border-line">
          <div className="mx-auto flex max-w-3xl items-baseline justify-between gap-6 px-5 py-4">
            <Link href="/" className="no-underline">
              <span className="eyebrow">Q-CHAT-10</span>
              <h1 className="mt-0.5 text-lg text-ink">Traits screening</h1>
            </Link>
            <nav className="flex gap-5 text-sm">
              <Link
                href="/"
                className="text-ink-soft no-underline hover:text-accent"
              >
                Questionnaire
              </Link>
              <Link
                href="/metrics"
                className="text-ink-soft no-underline hover:text-accent"
              >
                Model metrics
              </Link>
            </nav>
          </div>
        </header>
        {children}
        <footer className="mx-auto max-w-3xl px-5 py-10">
          <p className="text-xs leading-relaxed text-muted">
            A screening aid built on the public Q-CHAT-10 toddler dataset. It
            does not diagnose autism. A diagnosis comes from a full assessment
            by a paediatrician, developmental psychologist or psychiatrist. Use
            any result here only as a prompt to seek that assessment.
          </p>
        </footer>
      </body>
    </html>
  );
}
