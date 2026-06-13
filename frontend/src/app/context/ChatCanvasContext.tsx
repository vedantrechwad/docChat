"use client";
import React, { createContext, useContext, useState } from 'react';

type ChatMessage = {
  role: 'assistant' | 'user';
  content: string;
  sources?: any[];
};

type ChatCanvasContextType = {
  pendingCanvasItems: string[];
  addCanvasItem: (text: string) => void;
  clearCanvasItems: () => void;
  chatMessages: ChatMessage[];
  setChatMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
};

const ChatCanvasContext = createContext<ChatCanvasContextType>({
  pendingCanvasItems: [],
  addCanvasItem: () => {},
  clearCanvasItems: () => {},
  chatMessages: [],
  setChatMessages: () => {},
});

export const useChatCanvas = () => useContext(ChatCanvasContext);

export const ChatCanvasProvider = ({ children }: { children: React.ReactNode }) => {
  const [pendingCanvasItems, setPendingCanvasItems] = useState<string[]>([]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: "Hello! I'm your Study Companion. Upload some documents on the left, and then ask me questions, ask for summaries, or let me generate practice materials!"
    }
  ]);

  const addCanvasItem = (text: string) => {
    setPendingCanvasItems(prev => [...prev, text]);
  };

  const clearCanvasItems = () => {
    setPendingCanvasItems([]);
  };

  return (
    <ChatCanvasContext.Provider value={{ pendingCanvasItems, addCanvasItem, clearCanvasItems, chatMessages, setChatMessages }}>
      {children}
    </ChatCanvasContext.Provider>
  );
};
