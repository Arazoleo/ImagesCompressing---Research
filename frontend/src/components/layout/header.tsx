"use client";

import { Bell, Search, User, Moon, Sun, Command, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { useTheme } from "next-themes";
import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";
import { SidebarTrigger } from "@/components/ui/sidebar";

export function Header() {
  const { setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [searchFocused, setSearchFocused] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const toggleTheme = () => {
    document.documentElement.classList.add('transitioning');
    setTheme(resolvedTheme === 'dark' ? 'light' : 'dark');
    setTimeout(() => {
      document.documentElement.classList.remove('transitioning');
    }, 500);
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/80 backdrop-blur-xl supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-16 items-center px-4 md:px-6">
        {/* Mobile Sidebar Trigger */}
        <SidebarTrigger className="mr-4 md:hidden" />
        
        {/* Search Bar */}
        <div className="mr-4 hidden md:flex flex-1 max-w-md">
          <motion.div 
            className="relative w-full"
            animate={{ scale: searchFocused ? 1.01 : 1 }}
            transition={{ duration: 0.2 }}
          >
            <div className="relative flex items-center">
              <Search className="absolute left-3.5 h-4 w-4 text-muted-foreground pointer-events-none" strokeWidth={1.5} />
              <Input
                placeholder="Buscar imagens, algoritmos..."
                className={cn(
                  "pl-10 pr-20 h-10 bg-accent/50 border-border rounded-xl transition-all duration-300",
                  "focus:bg-background focus:border-foreground/20 focus:ring-1 focus:ring-foreground/10",
                  "placeholder:text-muted-foreground/60"
                )}
                onFocus={() => setSearchFocused(true)}
                onBlur={() => setSearchFocused(false)}
              />
              <div className="absolute right-3 flex items-center gap-1 pointer-events-none">
                <kbd className="hidden sm:inline-flex h-5 items-center gap-1 rounded-md border border-border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
                  <Command className="h-2.5 w-2.5" />K
                </kbd>
              </div>
            </div>
          </motion.div>
        </div>

        <div className="flex flex-1 items-center justify-end space-x-2">
          {/* Mobile Search */}
          <Button variant="ghost" size="icon" className="md:hidden rounded-xl hover:bg-accent">
            <Search className="h-5 w-5" strokeWidth={1.5} />
          </Button>

          {/* Theme Toggle */}
          {mounted && (
            <motion.div whileTap={{ scale: 0.95 }}>
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleTheme}
                className="relative rounded-xl overflow-hidden group hover:bg-accent"
              >
                <AnimatePresence mode="wait">
                  {resolvedTheme === 'dark' ? (
                    <motion.div
                      key="moon"
                      initial={{ rotate: -90, opacity: 0 }}
                      animate={{ rotate: 0, opacity: 1 }}
                      exit={{ rotate: 90, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Moon className="h-5 w-5" strokeWidth={1.5} />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="sun"
                      initial={{ rotate: 90, opacity: 0 }}
                      animate={{ rotate: 0, opacity: 1 }}
                      exit={{ rotate: -90, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Sun className="h-5 w-5" strokeWidth={1.5} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </Button>
            </motion.div>
          )}

          {/* Notifications */}
          <motion.div whileTap={{ scale: 0.95 }}>
            <Button 
              variant="ghost" 
              size="icon" 
              className="relative rounded-xl hover:bg-accent group"
            >
              <Bell className="h-5 w-5 transition-transform group-hover:rotate-12" strokeWidth={1.5} />
              <span className="absolute -top-0.5 -right-0.5 flex h-4 w-4 items-center justify-center">
                <span className="absolute inline-flex h-full w-full rounded-full bg-foreground/20 animate-ping" />
                <span className="relative flex h-4 w-4 items-center justify-center rounded-full bg-foreground text-[9px] font-semibold text-background">
                  3
                </span>
              </span>
            </Button>
          </motion.div>

          {/* Divider */}
          <div className="h-6 w-px bg-border mx-2 hidden sm:block" />

          {/* User Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button 
                variant="ghost" 
                className="relative h-10 px-2 rounded-xl hover:bg-accent group"
              >
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="flex items-center gap-3"
                >
                  <Avatar className="h-8 w-8 rounded-lg border border-border group-hover:border-foreground/20 transition-colors">
                    <AvatarImage src="/avatars/user.png" alt="User" />
                    <AvatarFallback className="bg-foreground text-background font-medium text-sm rounded-lg">
                      LA
                    </AvatarFallback>
                  </Avatar>
                  <div className="hidden sm:flex flex-col items-start">
                    <span className="text-sm font-medium">Leonardo</span>
                    <span className="text-[10px] text-muted-foreground">Pro</span>
                  </div>
                </motion.div>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent 
              className="w-64 p-2 rounded-xl border-border bg-popover/95 backdrop-blur-xl" 
              align="end" 
              forceMount
            >
              <DropdownMenuLabel className="font-normal p-3">
                <div className="flex items-center gap-3">
                  <Avatar className="h-12 w-12 rounded-xl border border-border">
                    <AvatarImage src="/avatars/user.png" alt="User" />
                    <AvatarFallback className="bg-foreground text-background font-semibold text-lg rounded-xl">
                      LA
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-semibold leading-none">Leonardo Arazo</p>
                    <p className="text-xs leading-none text-muted-foreground">
                      leonardo@imagestudio.com
                    </p>
                    <Badge variant="secondary" className="w-fit mt-1.5 text-[10px] bg-foreground text-background border-0 rounded-md">
                      <Sparkles className="h-2.5 w-2.5 mr-1" />
                      Pro
                    </Badge>
                  </div>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator className="bg-border" />
              <DropdownMenuItem className="rounded-lg cursor-pointer hover:bg-accent transition-colors p-2.5">
                <User className="mr-2 h-4 w-4" strokeWidth={1.5} />
                Perfil
              </DropdownMenuItem>
              <DropdownMenuItem className="rounded-lg cursor-pointer hover:bg-accent transition-colors p-2.5">
                <Sparkles className="mr-2 h-4 w-4" strokeWidth={1.5} />
                Configurações
              </DropdownMenuItem>
              <DropdownMenuItem className="rounded-lg cursor-pointer hover:bg-accent transition-colors p-2.5">
                <Bell className="mr-2 h-4 w-4" strokeWidth={1.5} />
                Histórico
              </DropdownMenuItem>
              <DropdownMenuSeparator className="bg-border" />
              <DropdownMenuItem className="rounded-lg cursor-pointer hover:bg-destructive/10 text-destructive transition-colors p-2.5">
                Sair
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
}

function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(' ');
}
