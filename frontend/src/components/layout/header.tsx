"use client";

import { Bell, Search, User, Moon, Sun, Sparkles, Command } from "lucide-react";
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
  const { theme, setTheme, resolvedTheme } = useTheme();
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
    }, 300);
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/80 backdrop-blur-xl supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-16 items-center px-4 md:px-6">
        {/* Mobile Sidebar Trigger */}
        <SidebarTrigger className="mr-4 md:hidden" />
        
        {/* Search Bar */}
        <div className="mr-4 hidden md:flex flex-1 max-w-md">
          <motion.div 
            className={`relative w-full transition-all duration-300 ${
              searchFocused ? 'scale-105' : 'scale-100'
            }`}
            animate={{ scale: searchFocused ? 1.02 : 1 }}
          >
            <div className={`absolute inset-0 rounded-xl transition-all duration-300 ${
              searchFocused 
                ? 'bg-gradient-to-r from-primary/20 via-primary/10 to-primary/20 blur-xl' 
                : 'bg-transparent'
            }`} />
            <div className="relative flex items-center">
              <Search className="absolute left-3 h-4 w-4 text-muted-foreground pointer-events-none" />
              <Input
                placeholder="Buscar imagens, algoritmos..."
                className="pl-10 pr-20 h-10 bg-muted/50 border-border/50 rounded-xl focus:bg-background focus:border-primary/50 transition-all duration-300"
                onFocus={() => setSearchFocused(true)}
                onBlur={() => setSearchFocused(false)}
              />
              <div className="absolute right-3 flex items-center gap-1 pointer-events-none">
                <kbd className="hidden sm:inline-flex h-5 items-center gap-1 rounded border border-border/50 bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
                  <Command className="h-3 w-3" />K
                </kbd>
              </div>
            </div>
          </motion.div>
        </div>

        <div className="flex flex-1 items-center justify-end space-x-2 md:space-x-4">
          {/* Mobile Search */}
          <Button variant="ghost" size="icon" className="md:hidden rounded-xl hover:bg-primary/10">
            <Search className="h-5 w-5" />
          </Button>

          {/* Theme Toggle */}
          {mounted && (
            <motion.div
              whileTap={{ scale: 0.95 }}
              whileHover={{ scale: 1.05 }}
            >
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleTheme}
                className="relative rounded-xl overflow-hidden group hover:bg-primary/10"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-amber-400/20 to-orange-500/20 dark:from-indigo-500/20 dark:to-purple-600/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                <AnimatePresence mode="wait">
                  {resolvedTheme === 'dark' ? (
                    <motion.div
                      key="moon"
                      initial={{ rotate: -90, opacity: 0 }}
                      animate={{ rotate: 0, opacity: 1 }}
                      exit={{ rotate: 90, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Moon className="h-5 w-5 text-indigo-400" />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="sun"
                      initial={{ rotate: 90, opacity: 0 }}
                      animate={{ rotate: 0, opacity: 1 }}
                      exit={{ rotate: -90, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                    >
                      <Sun className="h-5 w-5 text-amber-500" />
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
              className="relative rounded-xl hover:bg-primary/10 group"
            >
              <Bell className="h-5 w-5 transition-transform group-hover:rotate-12" />
              <motion.span
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="absolute -top-0.5 -right-0.5 flex h-5 w-5 items-center justify-center"
              >
                <span className="absolute inline-flex h-full w-full rounded-full bg-primary/40 animate-ping" />
                <Badge
                  variant="destructive"
                  className="relative h-5 w-5 rounded-full p-0 text-[10px] flex items-center justify-center bg-gradient-to-br from-rose-500 to-pink-600 border-0"
                >
                  3
                </Badge>
              </motion.span>
            </Button>
          </motion.div>

          {/* User Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button 
                variant="ghost" 
                className="relative h-10 w-10 rounded-xl p-0 hover:bg-primary/10 group"
              >
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="relative"
                >
                  <div className="absolute -inset-0.5 rounded-xl bg-gradient-to-br from-primary to-primary/50 opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-sm" />
                  <Avatar className="relative h-9 w-9 rounded-lg border-2 border-border/50 group-hover:border-primary/50 transition-colors">
                    <AvatarImage src="/avatars/user.png" alt="User" />
                    <AvatarFallback className="bg-gradient-to-br from-primary to-primary/80 text-primary-foreground font-semibold rounded-lg">
                      LA
                    </AvatarFallback>
                  </Avatar>
                  <span className="absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full bg-emerald-500 border-2 border-background" />
                </motion.div>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent 
              className="w-64 p-2 glass rounded-xl border-border/50" 
              align="end" 
              forceMount
            >
              <DropdownMenuLabel className="font-normal p-3">
                <div className="flex items-center gap-3">
                  <Avatar className="h-12 w-12 rounded-xl border-2 border-primary/20">
                    <AvatarImage src="/avatars/user.png" alt="User" />
                    <AvatarFallback className="bg-gradient-to-br from-primary to-primary/80 text-primary-foreground font-semibold text-lg rounded-xl">
                      LA
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-semibold leading-none">Leonardo Arazo</p>
                    <p className="text-xs leading-none text-muted-foreground">
                      leonardo@imagestudio.com
                    </p>
                    <Badge variant="secondary" className="w-fit mt-1 text-[10px] bg-primary/10 text-primary border-0">
                      <Sparkles className="h-2.5 w-2.5 mr-1" />
                      Pro
                    </Badge>
                  </div>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator className="bg-border/50" />
              <DropdownMenuItem className="rounded-lg cursor-pointer hover:bg-primary/10 transition-colors p-2.5">
                <User className="mr-2 h-4 w-4" />
                Perfil
              </DropdownMenuItem>
              <DropdownMenuItem className="rounded-lg cursor-pointer hover:bg-primary/10 transition-colors p-2.5">
                <Sparkles className="mr-2 h-4 w-4" />
                Configurações
              </DropdownMenuItem>
              <DropdownMenuItem className="rounded-lg cursor-pointer hover:bg-primary/10 transition-colors p-2.5">
                <Bell className="mr-2 h-4 w-4" />
                Histórico
              </DropdownMenuItem>
              <DropdownMenuSeparator className="bg-border/50" />
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
