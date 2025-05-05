import React, { useState, useEffect, useRef } from 'react';

/**
 * PropertyAssistAI Chat Widget
 * 
 * This is a demo of the client-facing chat widget that connects to the PropertyAssistAI backend.
 * It showcases the fast response times and natural conversation flow.
 */
const ChatWidget = () => {
  // State management
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      type: 'bot', 
      content: 'Hello! I\'m PropertyAssistAI, your real estate assistant. How can I help you find your dream property today?',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [conversation, setConversation] = useState({
    id: 'demo-' + Math.random().toString(36).substring(2, 9),
    startTime: new Date()
  });
  const [responseMetrics, setResponseMetrics] = useState([]);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Mock API call with artificial delay (simulating < 1s response time)
  const mockApiCall = async (message) => {
    // Start timing
    const startTime = performance.now();
    
    // Simulate network delay (randomized but < 800ms)
    const delay = Math.floor(Math.random() * 400) + 200;
    await new Promise(resolve => setTimeout(resolve, delay));
    
    // End timing
    const endTime = performance.now();
    const responseTime = (endTime - startTime) / 1000;
    
    // Track response time metrics
    setResponseMetrics(prev => [...prev, responseTime]);
    
    // Generate appropriate responses based on user input
    const lowercaseMsg = message.toLowerCase();
    
    if (lowercaseMsg.includes('property') && lowercaseMsg.includes('price')) {
      return {
        content: 'We have several properties that might fit your needs. Our listings range from €280,000 for apartments to €750,000 for detached houses. Could you tell me more about what area you're interested in or your specific budget range?',
        responseTime: responseTime
      };
    } else if (lowercaseMsg.includes('area') || lowercaseMsg.includes('location')) {
      return {
        content: 'We have properties available in Central Munich, Schwabing, Bogenhausen, and several other desirable neighborhoods. Each area offers different amenities and price points. Is there a specific neighborhood you're interested in?',
        responseTime: responseTime
      };
    } else if (lowercaseMsg.includes('contact') || lowercaseMsg.includes('broker') || lowercaseMsg.includes('agent')) {
      return {
        content: 'I'd be happy to connect you with one of our expert brokers. Could you please provide your name and either an email address or phone number so they can reach out to you?',
        leadCreated: true,
        responseTime: responseTime
      };
    } else if (lowercaseMsg.includes('appointment') || lowercaseMsg.includes('viewing') || lowercaseMsg.includes('see')) {
      return {
        content: 'I can help you schedule a viewing! We have availability this week on Wednesday afternoon and Friday morning. When would be most convenient for you? Also, could you share your contact information so our agent can confirm the details?',
        responseTime: responseTime
      };
    } else if (lowercaseMsg.includes('thank')) {
      return {
        content: 'You're very welcome! I'm here to help with any other questions you might have about properties or the buying/renting process. Feel free to reach out anytime.',
        responseTime: responseTime
      };
    } else {
      return {
        content: 'Thank you for your message. To better assist you with finding the perfect property, could you share more details about what you're looking for? For example, are you interested in buying or renting, what size property, and in which areas?',
        responseTime: responseTime
      };
    }
  };

  // Handle sending messages
  const handleSend = async () => {
    if (input.trim() === '') return;
    
    // Add user message
    const userMessage = {
      id: messages.length + 1,
      type: 'user',
      content: input,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    
    try {
      // Call API (mocked)
      const response = await mockApiCall(input);
      
      // Add bot response
      const botMessage = {
        id: messages.length + 2,
        type: 'bot',
        content: response.content,
        timestamp: new Date(),
        responseTime: response.responseTime,
        leadCreated: response.leadCreated || false
      };
      
      // Simulate minimal typing delay for realistic feel
      setTimeout(() => {
        setMessages(prev => [...prev, botMessage]);
        setIsTyping(false);
      }, 300);
      
    } catch (error) {
      console.error('Error sending message:', error);
      setIsTyping(false);
      
      // Add error message
      setMessages(prev => [...prev, {
        id: messages.length + 2,
        type: 'bot',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        isError: true,
        timestamp: new Date()
      }]);
    }
  };

  // Handle key press (Enter to send)
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  // Calculate average response time
  const averageResponseTime = responseMetrics.length 
    ? (responseMetrics.reduce((a, b) => a + b, 0) / responseMetrics.length).toFixed(3) 
    : '0.000';

  return (
    <div className="flex flex-col h-full max-w-lg mx-auto bg-white shadow-lg rounded-lg overflow-hidden">
      {/* Header */}
      <div className="bg-blue-600 text-white px-4 py-3 flex justify-between items-center">
        <div>
          <h3 className="font-bold text-lg">PropertyAssistAI</h3>
          <p className="text-xs opacity-75">
            Conversation ID: {conversation.id}
          </p>
        </div>
        <div className="text-right text-xs">
          <p>Avg. Response: {averageResponseTime}s</p>
          <p>Conversations: 1</p>
        </div>
      </div>
      
      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-4 bg-gray-50" style={{ height: '400px' }}>
        {messages.map(message => (
          <div 
            key={message.id} 
            className={`mb-3 flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div 
              className={`rounded-lg px-4 py-2 max-w-xs ${
                message.type === 'user' 
                  ? 'bg-blue-500 text-white' 
                  : message.isError 
                    ? 'bg-red-100 text-red-800' 
                    : 'bg-gray-200 text-gray-800'
              }`}
            >
              <p>{message.content}</p>
              <div className="text-xs mt-1 opacity-75 flex justify-between">
                <span>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
                {message.type === 'bot' && message.responseTime && (
                  <span className="ml-2">
                    {message.responseTime.toFixed(3)}s
                  </span>
                )}
                {message.leadCreated && (
                  <span className="ml-2 text-green-700 font-bold">
                    Lead Created ✓
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="flex justify-start mb-3">
            <div className="bg-gray-200 text-gray-800 rounded-lg px-4 py-2">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input Area */}
      <div className="border-t border-gray-200 px-4 py-3 bg-white">
        <div className="flex items-center">
          <input
            type="text"
            className="flex-1 border border-gray-300 rounded-l-lg px-4 py-2 focus:outline-none focus:border-blue-500"
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <button
            className="bg-blue-600 hover:bg-blue-700 text-white rounded-r-lg px-4 py-2 focus:outline-none"
            onClick={handleSend}
          >
            Send
          </button>
        </div>
        <div className="text-xs text-gray-500 mt-1">
          PropertyAssistAI responds in under 1 second, syncs with your CRM, and helps qualify leads.
        </div>
      </div>

      {/* CSS for typing indicator */}
      <style jsx>{`
        .typing-indicator {
          display: flex;
          align-items: center;
        }
        .typing-indicator span {
          height: 8px;
          width: 8px;
          margin: 0 1px;
          background-color: #606060;
          border-radius: 50%;
          display: inline-block;
          animation: typing 1.4s ease-in-out infinite;
        }
        .typing-indicator span:nth-child(1) {
          animation-delay: 0s;
        }
        .typing-indicator span:nth-child(2) {
          animation-delay: 0.2s;
        }
        .typing-indicator span:nth-child(3) {
          animation-delay: 0.4s;
        }
        @keyframes typing {
          0% {
            transform: scale(1);
            opacity: 0.7;
          }
          50% {
            transform: scale(1.5);
            opacity: 1;
          }
          100% {
            transform: scale(1);
            opacity: 0.7;
          }
        }
      `}</style>
    </div>
  );
};

export default ChatWidget;