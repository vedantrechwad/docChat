import type { Metadata } from "next";
import "./globals.css";
import styles from "./layout.module.css";
import { LayoutDashboard, BookOpen, Layers, Target, Map, Mic, Settings, Search, Bell } from "lucide-react";
import Link from "next/link";
import { ChatCanvasProvider } from "./context/ChatCanvasContext";

export const metadata: Metadata = {
  title: "Study Companion AI",
  description: "Your ultimate AI-powered study companion with canvas, flashcards, and RAG.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <div className={styles.layoutContainer}>
          {/* Sidebar */}
          <aside className={styles.sidebar}>
            <div className={styles.sidebarHeader}>
              <div className={styles.logoIcon}>
                <Layers size={28} />
              </div>
              <span className={styles.logo}>
                Study<span className="gradient-text">Companion</span>
              </span>
            </div>
            
            <nav className={styles.nav}>
              <Link href="/" className={`${styles.navItem} ${styles.navItemActive}`}>
                <LayoutDashboard size={20} />
                <span>Dashboard</span>
              </Link>
              
              <Link href="/sources" className={styles.navItem}>
                <BookOpen size={20} />
                <span>Sources & Chat</span>
              </Link>
              
              <Link href="/canvas" className={styles.navItem}>
                <Map size={20} />
                <span>Canvas</span>
              </Link>
              
              <Link href="/flashcards" className={styles.navItem}>
                <Layers size={20} />
                <span>Flashcards</span>
              </Link>
              
              <Link href="/planner" className={styles.navItem}>
                <Target size={20} />
                <span>Study Planner</span>
              </Link>

              <Link href="/audio" className={styles.navItem}>
                <Mic size={20} />
                <span>Audio Guides</span>
              </Link>
            </nav>

            <div className={styles.sidebarFooter}>
              <Link href="/settings" className={styles.navItem}>
                <Settings size={20} />
                <span>Settings</span>
              </Link>
            </div>
          </aside>

          {/* Main Content */}
          <main className={styles.mainContent}>
            <header className={styles.header}>
              <div className={styles.headerTitle}>Overview</div>
              <div className={styles.headerActions}>
                <button className={styles.iconButton}>
                  <Search size={18} />
                </button>
                <button className={styles.iconButton}>
                  <Bell size={18} />
                </button>
              </div>
            </header>
            
            <div className={styles.pageContent}>
              <ChatCanvasProvider>
                {children}
              </ChatCanvasProvider>
            </div>
          </main>
        </div>
      </body>
    </html>
  );
}
