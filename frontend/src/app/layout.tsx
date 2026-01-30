import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { QueryProvider } from "@/components/providers/query-provider";
import { ThemeProvider } from "@/components/providers/theme-provider";
import { Toaster } from "@/components/ui/sonner";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { Header } from "@/components/layout/header";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "ImageStudio | Advanced Image Processing",
  description: "Professional image processing platform using linear algebra and artificial intelligence.",
  keywords: ["image processing", "linear algebra", "SVD", "compression", "AI"],
  authors: [{ name: "ImageStudio" }],
  openGraph: {
    title: "ImageStudio",
    description: "Advanced image processing with linear algebra",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="pt-BR" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange={false}
        >
          <QueryProvider>
            <SidebarProvider>
              <div className="flex min-h-screen w-full bg-background">
                {/* Subtle background pattern */}
                <div className="fixed inset-0 dot-pattern pointer-events-none opacity-50" />
                <div className="fixed inset-0 gradient-radial-mono pointer-events-none" />
                
                <AppSidebar />
                <div className="flex flex-1 flex-col relative">
                  <Header />
                  <main className="flex-1 overflow-y-auto relative">
                    <div className="page-enter">
                      {children}
                    </div>
                  </main>
                </div>
              </div>
              <Toaster 
                position="bottom-right"
                toastOptions={{
                  className: "glass border-border/50 font-sans",
                }}
              />
            </SidebarProvider>
          </QueryProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
