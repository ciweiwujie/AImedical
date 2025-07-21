<template>
	<div class="app-container">
		<!-- 左侧边栏 -->
		<div class="sidebar" :class="{ 'sidebar-collapsed': isSidebarCollapsed }">
			<div class="sidebar-header">
				<h2 v-if="!isSidebarCollapsed">医疗助手</h2>
				<button class="collapse-btn" @click="toggleSidebar">
					<i :class="isSidebarCollapsed ? 'el-icon-s-unfold' : 'el-icon-s-fold'"></i>
				</button>
			</div>

			<div class="sidebar-content" v-if="!isSidebarCollapsed">
				<button class="new-chat-btn" @click="createNewChat">
					<i class="el-icon-plus"></i>
					<span>新建对话</span>
				</button>

				<div class="history-list">
					<div v-for="(chat, index) in chatHistory" :key="index" class="history-item"
						:class="{ 'active': currentChatIndex === index }" @click="switchChat(index)">
						<i class="el-icon-chat-dot-round"></i>
						<span class="history-title">{{ chat.title || '新对话' }}</span>
						<button class="delete-btn" @click.stop="deleteChat(index)">
							<i class="el-icon-close"></i>
						</button>
					</div>
				</div>
			</div>
		</div>

		<!-- 主聊天区域 -->
		<div class="main-content">
			<div v-if="!currentChat" class="empty-state">
				<i class="el-icon-chat-line-round"></i>
				<h3>欢迎使用医疗问答助手</h3>
				<p>点击"新建对话"开始咨询</p>
			</div>

			<div v-else class="chat-container">
				<div class="chat-header">
					<h3>{{ currentChat.title || '新对话' }}</h3>
				</div>

				<div class="chat-messages" ref="messagesContainer">
					<div v-for="(message, index) in currentChat.messages" :key="index" class="message-wrapper"
						:class="message.sender">
						<ChatMessage :message="message" />
					</div>

					<div v-if="isLoading" class="loading-wrapper">
						<div class="message-bubble assistant">
							<div class="message-content">
								<i class="el-icon-loading"></i> 正在思考中...
							</div>
						</div>
					</div>
				</div>

				<div class="input-area">
					<div class="input-wrapper">
						<textarea v-model="newMessage" placeholder="输入您的问题..." @keyup.enter.exact="sendMessage" rows="1"
							ref="textarea" @input="adjustTextareaHeight" :disabled="isLoading"></textarea>
						<button class="send-btn" @click="sendMessage" :disabled="!newMessage.trim() || isLoading">
							<i class="el-icon-s-promotion"></i>
						</button>
					</div>
					<p class="input-hint">医疗助手仅供参考，如有不适请及时就医</p>
				</div>
			</div>
		</div>
	</div>
</template>

<script>
	import ChatMessage from './components/ChatMessage.vue'

	export default {
		name: 'App',
		components: {
			ChatMessage
		},
		data() {
			return {
				isSidebarCollapsed: false,
				newMessage: '',
				chatHistory: [],
				currentChatIndex: 0, // 改为初始化为0
				isLoading: false,
				error: null
			}
		},
		computed: {
			currentChat() {
				return this.chatHistory.length > 0 && this.currentChatIndex !== null ?
					this.chatHistory[this.currentChatIndex] :
					null
			}
		},
		created() {
			console.log('App created hook');
			// 如果没有历史记录，初始化一个空对话
			if (this.chatHistory.length === 0) {
				this.createNewChat();
			}
		},
		methods: {
			toggleSidebar() {
				this.isSidebarCollapsed = !this.isSidebarCollapsed
			},

			// 创建新对话
			createNewChat() {
				console.log('createNewChat called');
				const newChat = {
					title: `对话 ${this.chatHistory.length + 1}`,
					messages: [{
						sender: 'assistant',
						content: '您好！我是医疗问答助手，可以为您解答常见的健康问题。请注意，我的回答仅供参考，不能替代专业医生的诊断。',
						timestamp: new Date()
					}],
					createdAt: new Date()
				}

				this.chatHistory.unshift(newChat)
				this.currentChatIndex = 0
				this.scrollToBottom()
				console.log('chatHistory after create:', this.chatHistory);
				console.log('currentChatIndex:', this.currentChatIndex);
			},

			// 切换对话
			switchChat(index) {
				this.currentChatIndex = index
				this.scrollToBottom()
			},

			deleteChat(index) {
				if (this.chatHistory.length <= 1) {
					this.$message.warning('至少需要保留一个对话')
					return
				}

				this.$confirm('确定要删除这个对话吗?', '提示', {
					confirmButtonText: '确定',
					cancelButtonText: '取消',
					type: 'warning'
				}).then(() => {
					this.chatHistory.splice(index, 1)

					if (index === this.currentChatIndex) {
						// 确保切换到有效的对话
						this.currentChatIndex = this.chatHistory.length > 0 ? 0 : null
					} else if (index < this.currentChatIndex) {
						// 更新当前索引
						this.currentChatIndex--
					}

					this.$message.success('对话已删除')
				}).catch(() => {
					this.$message.info('已取消删除')
				})
			},

			// 发送消息
			async sendMessage() {
				if (!this.newMessage.trim() || !this.currentChat || this.isLoading) return

				const userMessage = {
					sender: 'user',
					content: this.newMessage,
					timestamp: new Date()
				}

				this.currentChat.messages.push(userMessage)

				if (this.currentChat.messages.length === 2) {
					this.currentChat.title = this.newMessage.slice(0, 20) +
						(this.newMessage.length > 20 ? '...' : '')
				}

				this.newMessage = ''
				this.adjustTextareaHeight()
				this.scrollToBottom()

				try {
					this.isLoading = true
					this.error = null

					// 修改为正确的请求格式
					const response = await this.$http.post('/predict', {
						question: userMessage.content
					})
					console.log(response)
					const assistantMessage = {
						sender: 'assistant',
						content: response.data.answer,
						timestamp: new Date()
					}

					this.currentChat.messages.push(assistantMessage)
				} catch (error) {
					console.error('API请求失败:', error)
					this.error = error

					const errorMessage = {
						sender: 'assistant',
						content: '抱歉，获取回答时出错了，请稍后再试。',
						timestamp: new Date()
					}

					this.currentChat.messages.push(errorMessage)

					const errorMsg = error.response && error.response.data && error.response.data.message ?
						error.response.data.message :
						error.message
					this.$message.error('请求失败: ' + errorMsg)
				} finally {
					this.isLoading = false
					this.scrollToBottom()
				}
			},

			// 获取最近的对话历史（供API使用）
			getRecentHistory() {
				if (!this.currentChat) return []

				// 只返回最近的3轮对话
				const messages = this.currentChat.messages.slice(-6) // 3轮(每轮2条)

				return messages.map(msg => ({
					role: msg.sender === 'user' ? 'user' : 'assistant',
					content: msg.content
				}))
			},

			// UI辅助方法
			scrollToBottom() {
				this.$nextTick(() => {
					const container = this.$refs.messagesContainer
					if (container) {
						container.scrollTop = container.scrollHeight
					}
				})
			},

			adjustTextareaHeight() {
				this.$nextTick(() => {
					const textarea = this.$refs.textarea
					if (textarea) {
						textarea.style.height = 'auto'
						textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`
					}
				})
			}
		}
	}
</script>

<style src="./styles.css"></style>