"use client";

import {
  BarChart3,
  Upload,
  Settings,
  Cpu,
  GitCompare,
  FileImage,
  Home,
  Zap,
  FileText,
  Sparkles,
  ChevronRight,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarFooter,
  SidebarHeader,
} from "@/components/ui/sidebar";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const navigation = [
  {
    title: "Principal",
    items: [
      {
        title: "Dashboard",
        url: "/",
        icon: Home,
        description: "Visão geral",
      },
    ],
  },
  {
    title: "Processamento",
    items: [
      {
        title: "Upload",
        url: "/upload",
        icon: Upload,
        description: "Enviar imagens",
        badge: "Novo",
      },
      {
        title: "Processar",
        url: "/process",
        icon: Cpu,
        description: "Aplicar algoritmos",
      },
      {
        title: "Comparar",
        url: "/compare",
        icon: GitCompare,
        description: "Análise comparativa",
      },
    ],
  },
  {
    title: "Análise",
    items: [
      {
        title: "Resultados",
        url: "/results",
        icon: BarChart3,
        description: "Métricas detalhadas",
      },
      {
        title: "Relatórios",
        url: "/reports",
        icon: FileText,
        description: "Exportar dados",
      },
    ],
  },
  {
    title: "Galeria",
    items: [
      {
        title: "Todas as Imagens",
        url: "/gallery",
        icon: FileImage,
        description: "Biblioteca completa",
      },
    ],
  },
];

const bottomNavigation = [
  {
    title: "Configurações",
    url: "/settings",
    icon: Settings,
  },
];

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <Sidebar className="border-r border-border/50">
      <SidebarHeader className="border-b border-border/50 p-4">
        <Link href="/" className="flex items-center gap-3 group">
          <motion.div 
            className="relative"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <div className="absolute -inset-1 rounded-xl bg-gradient-to-br from-primary via-primary/80 to-primary/60 opacity-75 blur-sm group-hover:opacity-100 transition-opacity" />
            <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-primary/80 shadow-lg">
              <Zap className="h-5 w-5 text-primary-foreground" />
            </div>
          </motion.div>
          <div className="flex flex-col">
            <span className="text-base font-bold tracking-tight bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text">
              ImageStudio
            </span>
            <span className="text-[10px] text-muted-foreground font-medium tracking-wide uppercase">
              Álgebra Linear & IA
            </span>
          </div>
        </Link>
      </SidebarHeader>

      <SidebarContent className="px-2 py-4">
        {navigation.map((group, groupIndex) => (
          <SidebarGroup key={group.title} className="mb-2">
            <SidebarGroupLabel className="px-3 text-[10px] font-semibold tracking-wider uppercase text-muted-foreground/70 mb-2">
              {group.title}
            </SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu className="space-y-1">
                {group.items.map((item, itemIndex) => {
                  const isActive = pathname === item.url;
                  return (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton
                        asChild
                        isActive={isActive}
                        className="group"
                      >
                        <Link href={item.url}>
                          <motion.div
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: (groupIndex * 0.1) + (itemIndex * 0.05) }}
                            className={cn(
                              "relative flex items-center gap-3 w-full px-3 py-2.5 rounded-xl transition-all duration-200",
                              isActive 
                                ? "bg-primary text-primary-foreground shadow-lg shadow-primary/25" 
                                : "hover:bg-muted/80 text-foreground"
                            )}
                          >
                            {/* Active indicator */}
                            {isActive && (
                              <motion.div
                                layoutId="activeIndicator"
                                className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 rounded-full bg-primary-foreground"
                                transition={{ type: "spring", stiffness: 500, damping: 30 }}
                              />
                            )}
                            
                            {/* Icon */}
                            <div className={cn(
                              "flex items-center justify-center w-8 h-8 rounded-lg transition-all duration-200",
                              isActive 
                                ? "bg-primary-foreground/20" 
                                : "bg-muted group-hover:bg-primary/10 group-hover:text-primary"
                            )}>
                              <item.icon className="h-4 w-4" />
                            </div>
                            
                            {/* Text */}
                            <div className="flex-1 min-w-0">
                              <span className="text-sm font-medium block truncate">
                                {item.title}
                              </span>
                              {item.description && (
                                <span className={cn(
                                  "text-[10px] block truncate transition-colors",
                                  isActive 
                                    ? "text-primary-foreground/70" 
                                    : "text-muted-foreground group-hover:text-foreground/60"
                                )}>
                                  {item.description}
                                </span>
                              )}
                            </div>
                            
                            {/* Badge */}
                            {item.badge && (
                              <Badge 
                                variant="secondary" 
                                className={cn(
                                  "text-[9px] px-1.5 py-0.5 font-medium",
                                  isActive 
                                    ? "bg-primary-foreground/20 text-primary-foreground border-0" 
                                    : "bg-primary/10 text-primary border-0"
                                )}
                              >
                                {item.badge}
                              </Badge>
                            )}
                            
                            {/* Arrow */}
                            <ChevronRight className={cn(
                              "h-3.5 w-3.5 opacity-0 -translate-x-2 transition-all duration-200",
                              "group-hover:opacity-100 group-hover:translate-x-0",
                              isActive && "opacity-100 translate-x-0"
                            )} />
                          </motion.div>
                        </Link>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  );
                })}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>

      <SidebarFooter className="border-t border-border/50 p-4">
        {/* Pro Banner */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="relative overflow-hidden rounded-xl p-4 mb-3"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-primary/20 via-primary/10 to-transparent" />
          <div className="absolute inset-0 noise opacity-30" />
          <div className="relative">
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="h-4 w-4 text-primary" />
              <span className="text-xs font-semibold text-primary">Upgrade Pro</span>
            </div>
            <p className="text-[10px] text-muted-foreground leading-relaxed mb-3">
              Desbloqueie algoritmos avançados e processamento em lote
            </p>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full py-2 px-3 text-xs font-medium rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Conhecer Planos
            </motion.button>
          </div>
        </motion.div>

        {/* Settings */}
        {bottomNavigation.map((item) => {
          const isActive = pathname === item.url;
          return (
            <Link key={item.title} href={item.url}>
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200",
                  isActive 
                    ? "bg-muted text-foreground" 
                    : "hover:bg-muted/80 text-muted-foreground hover:text-foreground"
                )}
              >
                <item.icon className="h-4 w-4" />
                <span className="text-sm font-medium">{item.title}</span>
              </motion.div>
            </Link>
          );
        })}

        {/* Version */}
        <div className="mt-3 px-3">
          <p className="text-[10px] text-muted-foreground/50">
            ImageStudio v2.0.0
          </p>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
