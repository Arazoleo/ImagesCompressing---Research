import type { Metadata } from "next";
import { Playfair_Display, DM_Sans } from "next/font/google";
import { QueryProvider } from "@/components/providers/query-provider";
import { ThemeProvider } from "@/components/providers/theme-provider";
import { Toaster } from "@/components/ui/sonner";
import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { Header } from "@/components/layout/header";
import "./globals.css";

const playfair = Playfair_Display({
  variable: "--font-playfair",
  subsets: ["latin"],
  display: "swap",
});

const dmSans = DM_Sans({
  variable: "--font-dm-sans",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "ImageStudio | Processamento de Imagens com Álgebra Linear",
  description: "Studio avançado para processamento de imagens usando técnicas de álgebra linear computacional e inteligência artificial.",
  keywords: ["processamento de imagens", "álgebra linear", "SVD", "compressão", "IA"],
  authors: [{ name: "ImageStudio" }],
  openGraph: {
    title: "ImageStudio",
    description: "Processamento avançado de imagens com álgebra linear",
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
        className={`${playfair.variable} ${dmSans.variable} font-sans antialiased`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange={false}
        >
          <QueryProvider>
            <SidebarProvider>
              <div className="flex min-h-screen w-full bg-background">
                <AppSidebar />
                <div className="flex flex-1 flex-col">
                  <Header />
                  <main className="flex-1 overflow-y-auto">
                    <div className="page-enter">
                      {children}
                    </div>
                  </main>
                </div>
              </div>
              <Toaster 
                position="bottom-right"
                toastOptions={{
                  className: "glass border-border",
                }}
              />
            </SidebarProvider>
          </QueryProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
