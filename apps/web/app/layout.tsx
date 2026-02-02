import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "SelfStudy",
  description: "Local-first lecture notebook builder",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
