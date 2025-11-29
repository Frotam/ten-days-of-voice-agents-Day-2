'use client';

import React from 'react';
import { ChatEntry } from '@/components/livekit/chat-entry';
import { useChatMessages } from '@/hooks/useChatMessages';
import { useChat } from '@livekit/components-react';

export function AdventurePanel({ className = '' }: { className?: string }) {
  const messages = useChatMessages();
  const { send, isSending } = useChat();

  const gmMessages = messages.filter((m) => m.from?.isLocal !== true);
  const playerMessages = messages.filter((m) => m.from?.isLocal === true);

  const locale = typeof navigator !== 'undefined' ? navigator.language : 'en-US';

  const onRestart = async () => {
    try {
      // Send a restart chat message to the assistant. The GM agent will receive this message and can respond accordingly.
      await send?.('Restart story');
    } catch (e) {
      console.warn('Failed to trigger restart', e);
    }
  };

  return (
    <aside
      aria-label="Adventure Panel"
      className={`pointer-events-auto fixed bottom-20 right-4 z-50 w-80 max-h-[60vh] overflow-auto rounded-lg bg-muted p-3 shadow-lg ${className}`}
    >
      <div className="flex items-center justify-between pb-2">
        <h3 className="text-sm font-semibold">Adventure</h3>
        <button
          title="Restart the story"
          onClick={onRestart}
          disabled={isSending}
          className="rounded bg-primary px-2 py-1 text-xs text-white disabled:opacity-60"
        >
          Restart
        </button>
      </div>

      <div className="space-y-3">
        <section>
          <h4 className="text-xs text-muted-foreground">Game Master</h4>
          <ul className="mt-2 space-y-1">
            {gmMessages.length === 0 && <li className="text-xs text-muted-foreground">No messages yet.</li>}
            {gmMessages.map((m) => (
              <ChatEntry
                key={m.id}
                locale={locale}
                timestamp={m.timestamp}
                message={m.message}
                messageOrigin={'remote'}
                hasBeenEdited={false}
                name={m.from?.identity ?? 'Game Master'}
                className="text-sm"
              />
            ))}
          </ul>
        </section>

        <section className="pt-2">
          <h4 className="text-xs text-muted-foreground">You (transcript)</h4>
          <ul className="mt-2 space-y-1">
            {playerMessages.length === 0 && <li className="text-xs text-muted-foreground">No transcribed speech yet.</li>}
            {playerMessages.map((m) => (
              <ChatEntry
                key={m.id}
                locale={locale}
                timestamp={m.timestamp}
                message={m.message}
                messageOrigin={'local'}
                hasBeenEdited={false}
                name={m.from?.identity ?? 'You'}
                className="text-sm"
              />
            ))}
          </ul>
        </section>
      </div>
    </aside>
  );
}
