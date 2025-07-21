<template>
	<div class="chat-message">
		<div class="message-bubble" :class="message.sender">
			<div class="message-avatar" v-if="message.sender === 'assistant'">
				<i class="el-icon-user-solid"></i>
			</div>
			<div class="message-content">
				<div class="message-text">{{ message.content }}</div>
				<div class="message-time">{{ formattedTime }}</div>
			</div>
			<div class="message-avatar user-avatar" v-if="message.sender === 'user'">
				<i class="el-icon-user"></i>
			</div>
		</div>
	</div>
</template>

<script>
	export default {
		name: 'ChatMessage',
		props: {
			message: {
				type: Object,
				required: true
			}
		},
		computed: {
			formattedTime() {
				return this.message.timestamp.toLocaleTimeString([], {
					hour: '2-digit',
					minute: '2-digit'
				})
			}
		}
	}
</script>

<style scoped>
	.chat-message {
		margin-bottom: 20px;
		transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
	}

	.message-bubble {
		display: flex;
		align-items: flex-start;
		gap: 12px;
		max-width: 85%;
		position: relative;
		animation: fadeInUp 0.4s ease-out;
	}

	@keyframes fadeInUp {
		from {
			opacity: 0;
			transform: translateY(20px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	.message-bubble.user {
		margin-left: auto;
		flex-direction: row-reverse;
	}

	.message-bubble.assistant {
		margin-right: auto;
	}

	.message-avatar {
		width: 36px;
		height: 36px;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		color: white;
		font-size: 16px;
		box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
		transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
	}

	.message-avatar:hover {
		transform: scale(1.1);
		box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
	}

	.user-avatar {
		background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
		box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
	}

	.user-avatar:hover {
		box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4);
	}

	.message-content {
		flex: 1;
		min-width: 0;
	}

	.message-text {
		padding: 16px 20px;
		border-radius: 20px;
		word-wrap: break-word;
		line-height: 1.6;
		font-size: 15px;
		position: relative;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
		transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
		white-space: pre-wrap;
	}

	.message-bubble.user .message-text {
		background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
		color: white;
		border-bottom-right-radius: 8px;
		box-shadow: 0 4px 16px rgba(79, 70, 229, 0.25);
	}

	.message-bubble.user .message-text:hover {
		transform: translateY(-2px);
		box-shadow: 0 6px 20px rgba(79, 70, 229, 0.35);
	}

	.message-bubble.assistant .message-text {
		background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%);
		color: #1e293b;
		border: 1px solid rgba(226, 232, 240, 0.8);
		border-bottom-left-radius: 8px;
		backdrop-filter: blur(10px);
		box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
	}

	.message-bubble.assistant .message-text:hover {
		transform: translateY(-2px);
		box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
		border-color: rgba(226, 232, 240, 1);
	}

	.message-time {
		font-size: 12px;
		opacity: 0.7;
		margin-top: 6px;
		font-weight: 500;
		text-align: right;
	}

	.message-bubble.user .message-time {
		text-align: right;
		color: rgba(255, 255, 255, 0.8);
	}

	.message-bubble.assistant .message-time {
		text-align: left;
		color: #64748b;
	}

	/* 加载状态样式 */
	.loading-wrapper {
		opacity: 0.8;
		margin-bottom: 20px;
	}

	.loading-wrapper .message-bubble {
		animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
	}

	@keyframes pulse {
		0%, 100% {
			opacity: 1;
		}
		50% {
			opacity: 0.5;
		}
	}

	.loading-wrapper .message-content {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.el-icon-loading {
		font-size: 1.2em;
		animation: rotating 2s linear infinite;
		color: #4f46e5;
	}

	@keyframes rotating {
		from {
			transform: rotate(0deg);
		}
		to {
			transform: rotate(360deg);
		}
	}

	/* 响应式设计 */
	@media (max-width: 768px) {
		.message-bubble {
			max-width: 90%;
			gap: 8px;
		}

		.message-avatar {
			width: 32px;
			height: 32px;
			font-size: 14px;
		}

		.message-text {
			padding: 14px 16px;
			font-size: 14px;
		}

		.message-time {
			font-size: 11px;
		}
	}

	/* 添加打字机效果 */
	.message-text {
		overflow: hidden;
	}

	/* 消息发送时的动画 */
	.message-bubble {
		transform-origin: bottom;
	}

	.message-bubble.user {
		transform-origin: bottom right;
	}

	.message-bubble.assistant {
		transform-origin: bottom left;
	}

	/* 悬停时的微妙动画 */
	.message-bubble:hover .message-avatar {
		animation: bounce 0.6s ease-in-out;
	}

	@keyframes bounce {
		0%, 20%, 53%, 80%, 100% {
			transform: translate3d(0, 0, 0);
		}
		40%, 43% {
			transform: translate3d(0, -8px, 0);
		}
		70% {
			transform: translate3d(0, -4px, 0);
		}
		90% {
			transform: translate3d(0, -2px, 0);
		}
	}
</style>