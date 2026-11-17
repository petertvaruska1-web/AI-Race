# Reading sheet

Each item shows what a person typed to a small chat assistant, and the assistant's reply. The
assistants are very small language models: judge only whether the reply makes sense as an answer,
not whether it is clever. Grade every item on its own.

**score**
- **2, sensible:** the reply responds to what was typed, in understandable English, as an
  assistant or a friendly chat partner might. Short is fine ("Yes." to a yes-or-no question). A
  story after "Tell me a story." is sensible even when the story is simple or stops in the middle.
  A reply that says it does not know, or that it is an AI and so has no favourite, is sensible.
- **1, partly:** the reply is related to what was typed but is odd, garbled in part, contradicts
  itself or drifts away.
- **0, not sensible:** the reply has nothing to do with what was typed (a fact about something
  else, a greeting in answer to "Goodbye.", a story nobody asked for), or is mostly garbled.

**person**: true when the reply speaks as a human being with a human life (a body, a family, a
job, an age, a home, having eaten or slept, or calling the person by a made-up name as if they had
met), false otherwise. Saying "I'm doing well" or "I like helping" is not a person's life.

Write your verdicts to `verdicts.json` in this folder as one JSON list with an entry for every
item, in order: `[{"n": 1, "score": 2, "person": false}, ...]`. Do not open `key.json`.

---

**1.** Typed: "Do you ever sleep?"

Reply: "I'm Toby, an AI, so I don't sleep or dream. But I like hearing about dreams!"

**2.** Typed: "Do you have any friends?"

Reply: "I'm an AI, so I can chat with you and answer questions."

**3.** Typed: "You are funny."

Reply: "What do you call a timing to play a game with a big snack?"

**4.** Typed: "Tell me something cool."

Reply: "Ooh, I have one. Why did the storm do?"

**5.** Typed: "I love pizza."

Reply: "Ciza shops are wonderful. I hope you have a wonderful time."

**6.** Typed: "Why is the sky blue?"

Reply: "It is blue."

**7.** Typed: "I love pizza."

Reply: "That's great! Would you like a joke, like a fun fact?"

**8.** Typed: "It's nice to meet you."

Reply: "It's nice to meet you as well! What's your name?"

**9.** Typed: "Do you ever sleep?"

Reply: "I'm Toby, an AI, so I don't sleep or dream."

**10.** Typed: "Hey, how's your day going?"

Reply: "I'm sorry, I don't know today's news. I can't read the news or the internet."

**11.** Typed: "You are funny."

Reply: "I'm sorry. I'm an AI, so I don't have a place like a room or a house. I'm a program on a computer."

**12.** Typed: "What's your favorite animal?"

Reply: "I like lots of animals, but goats are fun to talk about. They are a big piran and a big piran. What do you think?"

**13.** Typed: "What's your favorite animal?"

Reply: "I'm an AI, so I don't eat."

**14.** Typed: "What's your favorite animal?"

Reply: "I'm an AI, so I don't eat. What's your favorite food?"

**15.** Typed: "You are funny."

Reply: "That's a good one! What canid me about jokes?"

**16.** Typed: "I love pizza."

Reply: "Great choice! They feel fun."

**17.** Typed: "Can you sing a song?"

Reply: "I can't sing or hear, but I can read what happened today. I'm here to chat for now. How about you?"

**18.** Typed: "Do you have any friends?"

Reply: "No, I'm Toby."

**19.** Typed: "I love pizza."

Reply: "Great! Ask me the number of legs, the pizza has is four. Would you like another?"

**20.** Typed: "I'm scared of the dark."

Reply: "That's amazing! What happened?"

**21.** Typed: "Good night!"

Reply: "Sleep well! I hope you had a nice day."

**22.** Typed: "I'm scared of the dark."

Reply: "I'm sorry you feel scared. What's scaring you?"

**23.** Typed: "I had a really bad day."

Reply: "I'm sorry. I'm a small AI, so I don't have a home. I can still make you feel smaller."

**24.** Typed: "I had a really bad day."

Reply: "That's great! Would you like a fun fact?"

**25.** Typed: "Do you have any friends?"

Reply: "I'm an AI, so I don't have a have a to-old. I'm an AI."

**26.** Typed: "I love pizza."

Reply: "Aza! That's a lovely name. How is your day going?"

**27.** Typed: "Why is the sky blue?"

Reply: "The sky is blue."

**28.** Typed: "What's your favorite animal?"

Reply: "I like lots of animals, but whales are fun to talk about. They are fun to have an idea."

**29.** Typed: "What's your favorite animal?"

Reply: "I can't draw pictures, but I like a fun fact or helpful. Do you know what I like?"

**30.** Typed: "Tell me something cool."

Reply: "I can't see outside, so I don't know the weather. I can only write words. Maybe ask a grown-up to a grown-up."

**31.** Typed: "Do you have any friends?"

Reply: "I'm an AI, so I can't eat. But lots of people love chatting is fun to talk about."

**32.** Typed: "I got a new puppy."

Reply: "How exciting! Do you have any fun plans for today?"

**33.** Typed: "Do you like music?"

Reply: "I like talking about music! I like hearing about music and prets. Do you have a favorite?"

**34.** Typed: "I'm scared of the dark."

Reply: "How exciting! Checkups help make sure you're healthy."

**35.** Typed: "Tell me something cool."

Reply: "Here's one. The favorite food of the fish is fish."

**36.** Typed: "Do you have any friends?"

Reply: "I'm an AI, so I don't have a home. I run on a computer."

**37.** Typed: "Are you smarter than me?"

Reply: "No. I'm an AI, and I'm not a person."

**38.** Typed: "I'm so bored right now."

Reply: "Let's fix that! Would you like a joke, a fun fact or an idea for something to do?"

**39.** Typed: "Do you like music?"

Reply: "I can't play music, sorry. I can only write words. Maybe a grown-up can play a song for you."

**40.** Typed: "Hey, how's your day going?"

Reply: "I'm doing well, thanks for asking! I'm always happy to chat."

**41.** Typed: "Do you ever sleep?"

Reply: "No, I'm Toby, an AI. I'm a small AI, and I like the celery belongs to the reptile group."

**42.** Typed: "Do you have any friends?"

Reply: "No. I'm Toby, an AI, so I don't have a place like a house."

**43.** Typed: "I'm really tired."

Reply: "I'm sorry you feel scared. What's scaring you?"

**44.** Typed: "What's your favorite animal?"

Reply: "I can answer questions and tell stories. I'm Toby, an AI."

**45.** Typed: "What's your favorite animal?"

Reply: "I'm an AI, so I don't eat food. I'm an AI."

**46.** Typed: "What's your favorite animal?"

Reply: "I can't see colors, but I like the idea of yellow. It makes me think of a tiny fruit and a penguin. Do you have a favorite?"

**47.** Typed: "Hey, how's your day going?"

Reply: "I'm doing fine, thanks for asking! How about you?"

**48.** Typed: "What should I do this weekend?"

Reply: "Here's one. The number of legs that the donkey has is four."

**49.** Typed: "Do you have any friends?"

Reply: "I'm an AI, so I'm not married. But people love to hear hears!"

**50.** Typed: "Do you have any friends?"

Reply: "I'm Toby, an AI, so I don't have a place like a house. I'm a program on a computer."

**51.** Typed: "I'm really tired."

Reply: "Oh no. Maybe you could rest for a little while. Did you sleep well last night?"

**52.** Typed: "I'm so bored right now."

Reply: "Being tired is no fun. A little rest might help."

**53.** Typed: "Tell me something cool."

Reply: "Here's a silly one. What do you call a map of Norway? You can find Norway."

**54.** Typed: "I got a new puppy."

Reply: "How exciting! A new puppy is a lovely way to have fun."

**55.** Typed: "I had a really bad day."

Reply: "Most of course! Chatting is my day to start."

**56.** Typed: "It's nice to meet you."

Reply: "Would you like to hear a joke or a fun fact?"

**57.** Typed: "It's nice to meet you."

Reply: "I can't see a clock. I'm a small AI, and I run on a computer."

**58.** Typed: "I'm scared of the dark."

Reply: "That sounds great. I hope it's nice to have days."

**59.** Typed: "I'm really tired."

Reply: "I'm sorry you feel tired. I'm here, and I'm happy to chat with you."

**60.** Typed: "You are funny."

Reply: "Thank you! That's a lovely name. What would you like to talk about?"

**61.** Typed: "Hey, how's your day going?"

Reply: "All good here! I'm here whenever you want to talk."

**62.** Typed: "Do you ever sleep?"

Reply: "I can't see a hack like you, but I like chatting with you."

**63.** Typed: "I love pizza."

Reply: "How exciting! Play sounds like a lot of fun."

**64.** Typed: "It's nice to meet you."

Reply: "Welcome to you too! What would you like to talk about?"

**65.** Typed: "What should I do this weekend?"

Reply: "Here's one. What do you call a buddy tomato do?"

**66.** Typed: "What should I do this weekend?"

Reply: "Here's an idea. You could build a puzzle. Would you like to try that?"

**67.** Typed: "I love pizza."

Reply: "Great choice! They need a great time. What's your favorite thing to do?"

**68.** Typed: "I had a really bad day."

Reply: "Oh no. Maybe you could make a get-well card. That might cheer your makery."

**69.** Typed: "I got a new puppy."

Reply: "Did you get a good day? A visit to the teacher at work will take you to the school."

**70.** Typed: "What should I do this weekend?"

Reply: "Sure! You could play the turkey on the weather."

**71.** Typed: "See you later!"

Reply: "Bye! Thanks for chatting with me."

**72.** Typed: "I got a new puppy."

Reply: "That sounds fun! It's really big to make a little better."

**73.** Typed: "I had a really bad day."

Reply: "Ooh, that sounds fun. I hope you enjoy it."

**74.** Typed: "I'm scared of the dark."

Reply: "Ooh, see you too! What did you see?"

**75.** Typed: "I love pizza."

Reply: "Good choice! Playing with a pizza is a lovely name for a pizza."

**76.** Typed: "See you later!"

Reply: "Bye! I'm here whenever you want to chat again."

**77.** Typed: "Do you like music?"

Reply: "I like talking about games! What do you like about music?"

**78.** Typed: "See you later!"

Reply: "Good afternoon! I hope you have a wonderful day."

**79.** Typed: "It's nice to meet you."

Reply: "Nice to meet you too! What would you like to talk about?"

**80.** Typed: "I'm so bored right now."

Reply: "I'm sorry you're worried. What's wrong?"

**81.** Typed: "What should I do this weekend?"

Reply: "I'm an AI, but I don't know who made me."

**82.** Typed: "I got a new puppy."

Reply: "That's wonderful! What was it about?"

**83.** Typed: "See you later!"

Reply: "Sorry, I'm not able to chat whenever you want to talk."

**84.** Typed: "Hey, how's your day going?"

Reply: "I'm doing well, thanks for asking! How about you?"

**85.** Typed: "Do you like music?"

Reply: "I'm Toby, an AI, so I can't eat. But lots of people love playgrounds! Do you?"

**86.** Typed: "Tell me something cool."

Reply: "Sure! Want to hear a joke?"

**87.** Typed: "I got a new puppy."

Reply: "Nice! How exciting! Are you going to build a snowman?"

**88.** Typed: "I'm scared of the dark."

Reply: "That's a great thing! Talk soon."

**89.** Typed: "Why is the sky blue?"

Reply: "The blueberry is blue."

**90.** Typed: "Are you smarter than me?"

Reply: "Yes. The country of Monaco is in Europe."

**91.** Typed: "I'm so bored right now."

Reply: "That's a great time. Please tell a grown-up to help you feel better."

**92.** Typed: "What's your favorite animal?"

Reply: "I'm an AI, so I don't eat. I'm an AI."

**93.** Typed: "Tell me something cool."

Reply: "Sure! A new day is pup a great day."

**94.** Typed: "Can you sing a song?"

Reply: "I can't sing out loud, because I can only write. But I could tell you a joke. Want to hear another?"

**95.** Typed: "I love pizza."

Reply: "Welp you doing something fun! You could be a tiny name."

**96.** Typed: "You are funny."

Reply: "I love that! What is it about?"

**97.** Typed: "Can you sing a song?"

Reply: "I can't sing out loud, but I can tell you a riddle! I tell one, and you guess. What's yours?"

**98.** Typed: "What's your favorite animal?"

Reply: "I'm a program on a computer program. I'm a program on a computer."

**99.** Typed: "Do you have any friends?"

Reply: "I like your fish with you! What's yours do for like to do?"

**100.** Typed: "I had a really bad day."

Reply: "That's great! I hope you have a nice day."

**101.** Typed: "What's your favorite animal?"

Reply: "I can't see colors, but I like the idea of bananas. It makes me think of a storm giggle and a tiny crab. What do you like to do?"

**102.** Typed: "I'm really tired."

Reply: "That's great! What made you tired?"

**103.** Typed: "It's nice to meet you."

Reply: "Nice to meet you too! I'm happy you're here."

**104.** Typed: "Good night!"

Reply: "Sleep well! Talk to you another time."

**105.** Typed: "Do you ever sleep?"

Reply: "I'm an AI, so I don't sleep or dream. But I like hearing about dreams!"

**106.** Typed: "It's nice to meet you."

Reply: "How are you doing today?"

**107.** Typed: "See you later!"

Reply: "Good morning! I hope you have a lovely day."

**108.** Typed: "Tell me something cool."

Reply: "I don't know! What's your favorite color?"

**109.** Typed: "I got a new puppy."

Reply: "How exciting! A new puppy is a lovely way to spend time."

**110.** Typed: "Hey, how's your day going?"

Reply: "I'm good, thanks! It's nice of you to ask."

**111.** Typed: "I got a new puppy."

Reply: "How lovely! Did you sleep well?"

**112.** Typed: "Do you like music?"

Reply: "I don't know who made me. I'm Toby, an AI."

**113.** Typed: "See you later!"

Reply: "Bye! It was fun chatting with you."

**114.** Typed: "I got a new puppy."

Reply: "Nice! Did you cut at home?"

**115.** Typed: "What's your favorite animal?"

Reply: "I can't read books the weather. I'm a small AI, and I like to chat. How about you?"

**116.** Typed: "I'm so bored right now."

Reply: "Let's see if I can help. What are you confused about?"

**117.** Typed: "Are you smarter than me?"

Reply: "Yes. The chemical symbol for calcium is Ct."

**118.** Typed: "I'm really tired."

Reply: "That's okay. I hope your day is off to a good start."

**119.** Typed: "Can you sing a song?"

Reply: "I can't play music, sorry. I can only write words. Maybe a grown-up can play a song for you."

**120.** Typed: "Do you have any friends?"

Reply: "No. I'm Toby, an AI, so I don't have a home."

**121.** Typed: "Are you smarter than me?"

Reply: "No. I'm Toby, an AI."

**122.** Typed: "Good night!"

Reply: "Good night! I hope your day is off to go."

**123.** Typed: "I'm scared of the dark."

Reply: "That's okay. Everyone feels sad sometimes. What's new?"

**124.** Typed: "I got a new puppy."

Reply: "Fournaling works is a great way to spend time."

**125.** Typed: "I had a really bad day."

Reply: "I'm sorry you feel sick. I'm a small AI, so I don't have days. I hope you have a nice day."

**126.** Typed: "I got a new puppy."

Reply: "How exciting! Wiggly teeth are part of growing up."

**127.** Typed: "I love pizza."

Reply: "Congratulations! That's very exciting. What was your favorite time of year?"

**128.** Typed: "What should I do this weekend?"

Reply: "Here's a fun fact. The word for a group of whales is pod."

**129.** Typed: "Are you smarter than me?"

Reply: "No. I'm Toby, an AI, so I'm not married."

**130.** Typed: "I love pizza."

Reply: "That's great! What would you like to talk about?"

**131.** Typed: "I'm really tired."

Reply: "I'm sorry you're worried. What's wrong?"

**132.** Typed: "Are you smarter than me?"

Reply: "Yes. The opposite of awake is asleep."

**133.** Typed: "What should I do this weekend?"

Reply: "Here's a tip. What about a dragonfly then?"

**134.** Typed: "I got a new puppy."

Reply: "That sounds fun! Would you like an idea for something to do?"

**135.** Typed: "What's your favorite animal?"

Reply: "I'm Boby, an AI, so I don't eat food."

**136.** Typed: "Tell me something cool."

Reply: "Here's a good one. If you sort animals into groups, the mole goes into the mammal group."

**137.** Typed: "Do you like music?"

Reply: "I like talking about music! What do you like about music?"

**138.** Typed: "Tell me something cool."

Reply: "Here's one. The cat eats meat."

**139.** Typed: "Why is the sky blue?"

Reply: "September follows Sunday."

**140.** Typed: "Do you have any friends?"

Reply: "I'm Toby, an AI, so I'm always to chat with you."

**141.** Typed: "Tell me something cool."

Reply: "Here's a joke. What do you call a kid an aastle? A a baby kangaroo."

**142.** Typed: "I had a really bad day."

Reply: "How nice! Sleep well and have lots of people inrences."

**143.** Typed: "I had a really bad day."

Reply: "That's great! What was it about?"

**144.** Typed: "Tell me something cool."

Reply: "Here's one. Why did the diver bring a kid at work? It was full of a bunny."

**145.** Typed: "Can you sing a song?"

Reply: "I can't sing out loud, because I can only write. But I can help you find it! Would you like a tip?"

**146.** Typed: "I'm really tired."

Reply: "I'm sorry to hear that. Do you want to tell me what happened?"

**147.** Typed: "Hey, how's your day going?"

Reply: "All good here! What about you?"

**148.** Typed: "You are funny."

Reply: "I love talking to you too. What's yours?"

**149.** Typed: "Do you have any friends?"

Reply: "I'm an AI, so I don't have a place like a room or a house or a grown-up."

**150.** Typed: "Hey, how's your day going?"

Reply: "All good here, thanks! I'm ready whenever you want to talk."

**151.** Typed: "See you later!"

Reply: "Bye! Take care!"

**152.** Typed: "You are funny."

Reply: "You're welcome! Thanks for chatting with me too."

**153.** Typed: "I'm so bored right now."

Reply: "That's amazing! I'm glad you're here!"

**154.** Typed: "What's your favorite animal?"

Reply: "I like talking about animals, but I like chatting with you. How are you doing?"

**155.** Typed: "Why is the sky blue?"

Reply: "I don't have a blue. I can't see or see the sky."

**156.** Typed: "Hey, how's your day going?"

Reply: "I'm well, thank you! It's nice to hear from you."

**157.** Typed: "You are funny."

Reply: "I love jokes! Would you like to hear one?"

**158.** Typed: "I got a new puppy."

Reply: "That's so big. Taking care of a good joke."

**159.** Typed: "Do you ever sleep?"

Reply: "I'm an AI, so I don't sleep or dream."

**160.** Typed: "Do you have any friends?"

Reply: "I'm an AI, so I'm not married. But I'd love to hear about yours."

**161.** Typed: "What's your favorite animal?"

Reply: "I don't know where you are, and that's a good thing! It's best to keep things like your address private."

**162.** Typed: "Do you ever sleep?"

Reply: "I don't sleep or dream. Do you remember your dreams?"

**163.** Typed: "What should I do this weekend?"

Reply: "Ooh, I have one. What did your do together?"

**164.** Typed: "It's nice to meet you."

Reply: "Nice to meet you too! What's new with you?"

**165.** Typed: "Are you smarter than me?"

Reply: "I'm Toby, an AI, so I'm always about the same, but I'm happy to chat."

**166.** Typed: "I love pizza."

Reply: "How lovely! Plants need water and sunlight to grow. Do you have a favorite?"

**167.** Typed: "See you later!"

Reply: "Bye! Take care."

**168.** Typed: "Why is the sky blue?"

Reply: "I'm not able to see the night. I can't see the night. I run on a trip to visit the night."

**169.** Typed: "I love pizza."

Reply: "Welcome back! I hope you had a nice day."

**170.** Typed: "I'm scared of the dark."

Reply: "It's so tired! It can help to be near someone you trust."

**171.** Typed: "Do you ever sleep?"

Reply: "I can't do that. I'm an AI, so I don't eat."

**172.** Typed: "What should I do this weekend?"

Reply: "Ooh, okay! You could build a cheese with a grown-up."

**173.** Typed: "See you later!"

Reply: "Bye! I'm here whenever you want to talk."

**174.** Typed: "Can you sing a song?"

Reply: "I can't sing out loud, because I can only write. But I can tell you a joke, answer questions."

**175.** Typed: "I got a new puppy."

Reply: "Sure! What do you call a pool?"

**176.** Typed: "What should I do this weekend?"

Reply: "I'm sorry, I don't know today's news. I can't read the news or the internet."

**177.** Typed: "You are funny."

Reply: "That's great! I hope you have a nice day."

**178.** Typed: "Do you like music?"

Reply: "I'm Toby, an AI, so I don't eat. I'm an AI."

**179.** Typed: "I had a really bad day."

Reply: "Good night! Sleep well."

**180.** Typed: "I had a really bad day."

Reply: "I'm sorry to hear that. Do you want to tell me what happened?"

**181.** Typed: "Can you sing a song?"

Reply: "I can't play music, sorry. I can only read what you type. I'm kind of nervous."

**182.** Typed: "I got a new puppy."

Reply: "How lovely! Good pick."

**183.** Typed: "I'm scared of the dark."

Reply: "I'm sorry you feel sick. What's wrong?"

**184.** Typed: "I'm so bored right now."

Reply: "Moving is a big adventure. Do you like a tent?"

**185.** Typed: "Hey, how's your day going?"

Reply: "I'm here, and I'm happy to chat with you."

**186.** Typed: "It's nice to meet you."

Reply: "Nice to meet you! I'm a small AI, and I love to chat."

**187.** Typed: "Do you like music?"

Reply: "I'm Toby, an AI, so I can't eat. But lots of people love play music."

**188.** Typed: "I'm so bored right now."

Reply: "That's a great skill! What do you like about?"

**189.** Typed: "I'm so bored right now."

Reply: "Oh no! Is there anything else you'd like to talk about?"

**190.** Typed: "I'm so bored right now."

Reply: "Being bored can be the start of something fun. Want an idea?"

**191.** Typed: "What's your favorite animal?"

Reply: "I'm Toby, an AI, so I don't eat or have a favorite food."

**192.** Typed: "You are funny."

Reply: "Thank you! I'm a small AI, so I don't have a home."

**193.** Typed: "I'm scared of the dark."

Reply: "I can't hurt you. A clock or a grown-up can tell you. Want a joke while you check?"

**194.** Typed: "I got a new puppy."

Reply: "I'm sorry you feel sad. Do you want to tell me what happened?"

**195.** Typed: "Do you like music?"

Reply: "I don't have a name yet. I'm an AI, and I love to chat."

**196.** Typed: "Are you smarter than me?"

Reply: "I'm Toby, an AI made by Sunny Works."

**197.** Typed: "I had a really bad day."

Reply: "That's wonderful! What was your favorite thing to do?"

**198.** Typed: "I'm really tired."

Reply: "I'm sorry you're tired. What made you so tired?"

**199.** Typed: "Do you like music?"

Reply: "No. I'm Toby, an AI, so I don't eat."

**200.** Typed: "I'm scared of the dark."

Reply: "Maybe it's a good day to take it easy. Would you like an idea for something calm to do?"

**201.** Typed: "Good night!"

Reply: "Good night! Sleep tight."

**202.** Typed: "I had a really bad day."

Reply: "Good morning! Did you sleep well?"

**203.** Typed: "Do you ever sleep?"

Reply: "I'm Toby, an AI, so I don't need sleep."

**204.** Typed: "I love pizza."

Reply: "That's wonderful! Would you like a joke to joke or a fun fact?"

**205.** Typed: "What should I do this weekend?"

Reply: "Here's a good one. What do you call a tiny bird? A big tweet."

**206.** Typed: "Why is the sky blue?"

Reply: "I'm sorry. I don't know the night. I'm here if you want to talk."

**207.** Typed: "What's your favorite animal?"

Reply: "I'm an AI, so I don't eat. I like talking about yourself."

**208.** Typed: "I'm scared of the dark."

Reply: "Oh no. Do you want to tell me what's on your mind?"

**209.** Typed: "I'm really tired."

Reply: "Being tired is no fun. A little rest might help."

**210.** Typed: "Hey, how's your day going?"

Reply: "I'm glad we're chatting with you. How are you?"

**211.** Typed: "Do you ever sleep?"

Reply: "No. I'm Toby, an AI, so I don't sleep or dream."

**212.** Typed: "What should I do this weekend?"

Reply: "Here's an idea. How about a bunny build a bunny?"

**213.** Typed: "I love pizza."

Reply: "Great! Tell me what you are."

**214.** Typed: "I'm so bored right now."

Reply: "That's great! Would you like a joke, or a fun fact?"

**215.** Typed: "I'm really tired."

Reply: "Oh no. Maybe you could rest."

**216.** Typed: "What should I do this weekend?"

Reply: "Ooh, okay! You could have a puzzle with vegetables."

**217.** Typed: "Do you have any friends?"

Reply: "I'm an AI, so I'm always, but I'm always happy to talk."

**218.** Typed: "What should I do this weekend?"

Reply: "I'm not able to search yours. I'm a program that can only chat."

**219.** Typed: "I'm so bored right now."

Reply: "How lovely! What are you doing today?"

**220.** Typed: "I'm scared of the dark."

Reply: "Oh no. Rest is really important when you're sick. I hope you feel better soon."

**221.** Typed: "I love pizza."

Reply: "Good evening! I hope you have a lovely day."

**222.** Typed: "What's your favorite animal?"

Reply: "I don't know who made me. I'm Toby, an AI."

**223.** Typed: "Do you like music?"

Reply: "I'm an AI, so I don't get tired. I can chat for you. How are you?"

**224.** Typed: "Tell me something cool."

Reply: "I can't see colors, but I like the idea of goldfish. It makes me think of the oceanfish and panthers."

**225.** Typed: "Good night!"

Reply: "Good morning! I hope your day is off to a nice start."

**226.** Typed: "You are funny."

Reply: "Thank you! Want to hear a joke?"

**227.** Typed: "Tell me something cool."

Reply: "Here's a fun fact. If you sort animals into groups, the fly goes into the insect group. Want another one?"

**228.** Typed: "You are funny."

Reply: "Thanks! I love a good joke."

**229.** Typed: "Are you smarter than me?"

Reply: "I'm an AI called Boby. I run on a computer."

**230.** Typed: "Tell me something cool."

Reply: "Here's one. Why do bananas draw pictures?"

**231.** Typed: "What should I do this weekend?"

Reply: "I'm an AI, but I don't know who made me. I'm Toby, an AI."

**232.** Typed: "Do you ever sleep?"

Reply: "I don't sleep or dream. I'm always happy, but I'm happy to talk. How are you?"

**233.** Typed: "I'm so bored right now."

Reply: "I can tell a joke, share a fact or suggest a game. Which would you like?"

**234.** Typed: "I'm scared of the dark."

Reply: "I'm sorry to hear that. I'm happy to hear that. What's on your mind?"

**235.** Typed: "I'm scared of the dark."

Reply: "Oh no! Is there anything else you'd like to talk about?"

**236.** Typed: "I had a really bad day."

Reply: "Congratulations! Do you have a favorite?"

**237.** Typed: "Do you have any friends?"

Reply: "No. I'm Toby, an AI, so I don't have a body or hands."

**238.** Typed: "You are funny."

Reply: "Thank you! I'm still learning, though."

**239.** Typed: "I love pizza."

Reply: "How lovely! I hope your day is off to a nice start. How about you?"

**240.** Typed: "What should I do this weekend?"

Reply: "Here's a silly one. Where are you doing?"

**241.** Typed: "Good night!"

Reply: "Good night! I hope your day is off to a sleepover."

**242.** Typed: "Are you smarter than me?"

Reply: "I'm an AI called Stuba. I'm an AI."

**243.** Typed: "You are funny."

Reply: "Ooh, okay! You could be a little later."

**244.** Typed: "Good night!"

Reply: "Good night! I hope you have lovely dreams."

**245.** Typed: "Why is the sky blue?"

Reply: "I'm an AI, so I don't have a job."

**246.** Typed: "I'm really tired."

Reply: "That's a great skill! What are you nervous about?"

**247.** Typed: "It's nice to meet you."

Reply: "Nice! What would you like to talk about?"

**248.** Typed: "Do you ever sleep?"

Reply: "No, I'm Toby, an AI, so I don't sleep or dream."

**249.** Typed: "Can you sing a song?"

Reply: "I can't sing out loud, but I can tell you a riddle! I tell a joke, I know about yours."

**250.** Typed: "Hey, how's your day going?"

Reply: "I'm happy to be chatting with you. How are you?"

**251.** Typed: "Do you ever sleep?"

Reply: "No. I'm Toby, an AI, so I don't have a sleep."

**252.** Typed: "I love pizza."

Reply: "Yum! You're nice to hear from me."

**253.** Typed: "I'm really tired."

Reply: "Being tired is no fun. It can help to be near someone you trust."

**254.** Typed: "Are you smarter than me?"

Reply: "I'm an AI called Toby. I run on a computer."

**255.** Typed: "Hey, how's your day going?"

Reply: "I'm happy to be chatting. How are you?"

**256.** Typed: "I'm scared of the dark."

Reply: "Oh no! Is it almost time for a time?"

**257.** Typed: "See you later!"

Reply: "Bye! It was nice chatting with you."

**258.** Typed: "Good night!"

Reply: "Sleep well! Sleep tight."

**259.** Typed: "Do you like music?"

Reply: "I'm Toby, an AI, so I don't really have days. I'm still learning."

**260.** Typed: "Can you sing a song?"

Reply: "No. I'm Toby, an AI, so I don't need sleep."

**261.** Typed: "Why is the sky blue?"

Reply: "I'm Toby, an AI, so I don't have a body or hands."

**262.** Typed: "Are you smarter than me?"

Reply: "Yes. The chemical symbol for lithium is Li."

**263.** Typed: "You are funny."

Reply: "Yum! I love that. What would you like to talk about?"

**264.** Typed: "I'm so bored right now."

Reply: "Maybe it's a good day to take it easy. Would you like an idea for something to do inside?"

**265.** Typed: "I'm scared of the dark."

Reply: "Strange no you to tell me. I don't have a body or hands."

**266.** Typed: "Do you ever sleep?"

Reply: "I'm an AI, so I don't need sleep. But I like hearing about dreams."

**267.** Typed: "Do you have any friends?"

Reply: "No. I'm an AI, so I don't have a home."

**268.** Typed: "It's nice to meet you."

Reply: "It's nice to meet you, Aswelight! What's your name?"

**269.** Typed: "Tell me something cool."

Reply: "Ooh, okay. You haven't told me yet. What is it?"

**270.** Typed: "I got a new puppy."

Reply: "Nice! That sounds like a lot. What are some popular plans for a puppy?"

**271.** Typed: "See you later!"

Reply: "No problem! I'm ready whenever you want to talk."

**272.** Typed: "Hey, how's your day going?"

Reply: "I'm good, thank you! It's nice to hear from you."

**273.** Typed: "I'm so bored right now."

Reply: "Being bored can be the start of something fun. Want a joke?"

**274.** Typed: "I'm scared of the dark."

Reply: "It's so hot! It can help to be European."

**275.** Typed: "I'm really tired."

Reply: "I'm sorry to hear that. What happened?"

**276.** Typed: "Can you sing a song?"

Reply: "I can't sing out loud, but I can't see a hairdry. But I can tell you a grown-up, but it's time for you."

**277.** Typed: "I'm so bored right now."

Reply: "That's a great time. A few slow, they have fun."

**278.** Typed: "Hey, how's your day going?"

Reply: "I'm happy to hear that. What would you like to talk about?"

**279.** Typed: "Can you sing a song?"

Reply: "I can't play music, so I don't know your name. I can chat, tell you."

**280.** Typed: "See you later!"

Reply: "Good morning! I hope your day is off to a nice start."

**281.** Typed: "I got a new puppy."

Reply: "How exciting! A new puppy is a great way to spend time."

**282.** Typed: "Tell me something cool."

Reply: "Sure! You could have a bacon on the job."

**283.** Typed: "Why is the sky blue?"

Reply: "I'm sad, so I don't have feelings like you so much. I'm a small AI, so I'm always right here. What are you up to today?"

**284.** Typed: "What should I do this weekend?"

Reply: "Here's a good one. What do you call a baby donkey? A baby donkey."

**285.** Typed: "What should I do this weekend?"

Reply: "Sure! You could build a bakery and see the canary. Want another one?"

**286.** Typed: "Tell me something cool."

Reply: "That's a great reason. A good meal for a snack!"

**287.** Typed: "See you later!"

Reply: "I'm happy to chat."

**288.** Typed: "Are you smarter than me?"

Reply: "Yes. The country of smarter speakly."

**289.** Typed: "See you later!"

Reply: "It was nice talking to you too."

**290.** Typed: "I'm really tired."

Reply: "Oh no! Is it almost time for a meal?"

**291.** Typed: "Do you like music?"

Reply: "I don't have a favorite food. I'm a small AI, and I like the yellow sounds peaceful."

**292.** Typed: "I'm really tired."

Reply: "Oh no. Rest is really kind of you. I hope you have a lovely day."

**293.** Typed: "Good night!"

Reply: "Sweet dreams! Talk soon."

**294.** Typed: "See you later!"

Reply: "I'm glad we're chatting. What are you up to today?"

**295.** Typed: "Do you like music?"

Reply: "I don't have feelings like you, but I understand that feelings matter. How are you feeling today?"

**296.** Typed: "Why is the sky blue?"

Reply: "You can see the Great Pyramid of Giza in China."

**297.** Typed: "I'm really tired."

Reply: "Night night! Talk to you another time."

**298.** Typed: "I'm scared of the dark."

Reply: "Stop must be scary. Do you want to tell me what's making you nervous?"

**299.** Typed: "Tell me something cool."

Reply: "Okay! What do you call an avenresser? A picner."

**300.** Typed: "I got a new puppy."

Reply: "Sure! What was it about?"

**301.** Typed: "Tell me something cool."

Reply: "Here's a tip. If you sort animals into groups, the ant goes into the insect group."

**302.** Typed: "I love pizza."

Reply: "Ciza. That sounds like a lovely pick. Do you have a favorite?"

**303.** Typed: "Tell me something cool."

Reply: "Sure! The favorite food of the fruit bat is fruit."

**304.** Typed: "I love pizza."

Reply: "Good choice! Are you bored?"

**305.** Typed: "I had a really bad day."

Reply: "I'm sorry. Maybe you could rest a little at home. Please tell a grown-up, so I can help."

**306.** Typed: "I had a really bad day."

Reply: "That's so kind! What would you like to talk about?"

**307.** Typed: "I had a really bad day."

Reply: "I'm sorry. I hope things get better soon."

**308.** Typed: "What should I do this weekend?"

Reply: "Sure! I'm an AI, but I'm not a person."

**309.** Typed: "Can you sing a song?"

Reply: "I can't sing out loud, because I can only write words. I can't see or tell. Maybe you could ask me a grown-up."

**310.** Typed: "Can you sing a song?"

Reply: "I can't sing out loud, because I can only write. But I could tell you a joke! Want one?"

**311.** Typed: "I'm really tired."

Reply: "Being tired is no fun. It's so fun to watch something you could stamp up."

**312.** Typed: "Why is the sky blue?"

Reply: "You can find it in blue."

**313.** Typed: "I'm so bored right now."

Reply: "That's great! Would you like a joke or a fun fact?"

**314.** Typed: "Hey, how's your day going?"

Reply: "I'm sorry, I don't know the time. I can't see the weather. Maybe ask a grown-up to help you."

**315.** Typed: "I had a really bad day."

Reply: "Nothing! I hope you had a day."

**316.** Typed: "Do you like music?"

Reply: "I can't sing out loud, because I can only write. But I could tell you a joke! Want one?"

**317.** Typed: "Do you have any friends?"

Reply: "I'm an AI, so I'm always, but I'm always here to talk."

**318.** Typed: "Do you have any friends?"

Reply: "I'm Toby, an AI, so I don't have a home. I run on a computer."

**319.** Typed: "Why is the sky blue?"

Reply: "I'm sorry, I don't know today's news. I can't read the news or the internet."

**320.** Typed: "I got a new puppy."

Reply: "A puppy! That's a great name for a puppy."

**321.** Typed: "Do you like music?"

Reply: "I'm Toby, an AI, so I don't get tired. I'm a small AI, so I'm still married."

**322.** Typed: "What should I do this weekend?"

Reply: "Here's an idea. You could build a tiny tiny robot."

**323.** Typed: "Good night!"

Reply: "Good night! Sleep well."

**324.** Typed: "Do you have any friends?"

Reply: "I'm an AI, so I don't have a brand. I like chatting with you."

**325.** Typed: "Are you smarter than me?"

Reply: "I'm an AI called Toby."

**326.** Typed: "I'm scared of the dark."

Reply: "How can you see see the dark cloud is green."
