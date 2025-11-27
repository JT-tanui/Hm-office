# Tanui Assistant - Feature Roadmap

## 🎯 High-Impact Features

### 1. Voice Cloning 🎤
- Upload voice samples
- Train custom TTS voices using Supertonic
- Save custom voices to database
- Priority: **HIGH** (docs already exist)

### 2. Streaming Responses ⚡
- Stream LLM responses word-by-word (SSE/WebSockets)
- Show typing indicator while thinking
- Faster perceived response time
- Can interrupt mid-response
- Priority: **HIGH**

### 3. Search & Filter Conversations 🔍
- Full-text search across all messages
- Filter by date, model, conversation style
- SQLite FTS5 for fast searching
- Search results highlighting
- Priority: **HIGH**

### 4. Export Conversations 📤
- Export to Markdown, PDF, TXT, JSON
- Share conversations via link
- Import conversations from other apps
- Backup/restore functionality
- Priority: **MEDIUM**

---

## 🎨 UI/UX Improvements

### 5. Theme Customization 🌓
- Light/dark mode toggle
- Custom accent colors
- Font size adjustment
- Accessibility settings
- Priority: **MEDIUM**

### 6. Keyboard Shortcuts ⌨️
- `Ctrl+K` - New conversation
- `Ctrl+/` - Toggle settings
- `Ctrl+F` - Search
- `Ctrl+Shift+V` - Toggle voice input
- `Esc` - Exit Call Mode
- Priority: **LOW**

### 7. Markdown Editor ✍️
- Rich text input with formatting toolbar
- Syntax highlighting for code
- Live preview
- Code block copy button
- Priority: **MEDIUM**

### 8. Mobile-Responsive Design 📱
- Swipe gestures
- Bottom navigation bar
- Touch-optimized buttons
- Progressive Web App (PWA) support
- Priority: **HIGH**

---

## 🧠 Intelligence Features

### 9. Context-Aware Suggestions 💡
- Smart quick prompts based on context
- Auto-suggested follow-ups
- Related conversation recommendations
- Common task templates
- Priority: **MEDIUM**

### 10. Multi-Language Support 🌍
- Translate conversations
- Speak in multiple languages
- Auto-detect language
- Mixed language conversations
- Priority: **LOW**

### 11. Voice Activity Detection (VAD) 🎙️
- Better silence detection
- Background noise suppression
- Echo cancellation
- Mic sensitivity slider
- Priority: **MEDIUM**

---

## 📊 Organization & Productivity

### 12. Conversation Tags & Folders 🏷️
- Tag conversations (Work, Personal, Learning)
- Create custom folders
- Pin important conversations
- Archive old conversations
- Priority: **HIGH**

### 13. Conversation Analytics 📈
- Total messages sent/received
- Most used models
- Average response time
- Word count, conversation duration
- Usage trends over time
- Priority: **LOW**

### 14. Voice Memos 🎵
- Quick voice notes (no LLM response)
- Auto-transcribe memos
- Organize by date
- Export transcriptions
- Priority: **LOW**

---

## 🔧 Advanced Features

### 15. Multi-Modal Input 🖼️
- Upload images with questions
- Screenshot analysis
- PDF/document parsing
- Vision model integration
- Priority: **MEDIUM**

### 16. Custom Wake Words 🎯
- User-defined wake words
- Multiple wake word support
- Wake word sensitivity slider
- Custom activation sounds
- Priority: **LOW**

### 17. Integration Hub 🔌
- Google Calendar integration
- Email drafting
- Note-taking apps (Notion, Obsidian)
- Task managers (Todoist, Trello)
- Webhook support
- Priority: **LOW**

### 18. Voice Profiles 👥
- Multiple user profiles
- Voice recognition
- Per-user preferences
- Separate conversation histories
- Priority: **LOW**

---

## 🚀 Performance & Quality

### 19. Response Caching 💾
- Cache common responses
- Faster repeated questions
- Reduced API costs
- LRU cache strategy
- Priority: **MEDIUM**

### 20. Offline Mode 🔌
- Queue messages when offline
- Local-only Ollama fallback
- Sync when back online
- Offline conversation viewing
- Priority: **LOW**

---

## 🎁 Quick Wins (Batch 1 - Implementation Started)

### ✅ Implemented
- [x] Conversation styles
- [x] Voice speed control
- [x] Auto-continue timeout
- [x] Quick prompts
- [x] Wake word detection
- [x] SQLite database

### 🚧 In Progress (Batch 1)
- [ ] **Copy message button** - Copy assistant responses
- [ ] **Regenerate response** - Re-run last prompt
- [ ] **Edit & resend** - Edit user messages
- [ ] **Audio playback controls** - Pause, resume, speed
- [ ] **Typing indicators** - Show when assistant is thinking
- [ ] **Empty state graphics** - Better onboarding UI

---

## 📋 Implementation Priority

### Phase 7: Quick Wins (Current)
1. Copy button on messages
2. Regenerate response button
3. Edit message functionality
4. Audio playback controls
5. Typing indicator
6. Empty state UI

### Phase 8: Core Features
1. Voice Cloning
2. Streaming Responses
3. Search & Export

### Phase 9: Organization
1. Tags & Folders
2. Mobile PWA
3. Keyboard Shortcuts

### Phase 10: Advanced
1. Multi-modal input
2. Context-aware suggestions
3. Response caching

---

**Last Updated:** 2025-11-25
**Status:** Phase 7 (Quick Wins) in progress
