<template>
	<div class="chat-message">
		<div class="message-bubble" :class="message.sender">
			<div class="message-content">{{ message.content }}</div>
			<div class="message-time">{{ formattedTime }}</div>
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
		margin-bottom: 16px;
		transition: all 0.3s;
	}

	.message-bubble {
		max-width: 80%;
		padding: 12px 16px;
		border-radius: 12px;
		position: relative;
		word-wrap: break-word;
		line-height: 1.5;
		box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
		animation: fadeIn 0.3s ease;
	}

	@keyframes fadeIn {
		from {
			opacity: 0;
			transform: translateY(10px);
		}

		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	.message-bubble.user {
		background-color: #409EFF;
		color: white;
		margin-left: auto;
		border-bottom-right-radius: 4px;
	}

	.message-bubble.assistant {
		background-color: #f5f7fa;
		color: #333;
		margin-right: auto;
		border-bottom-left-radius: 4px;
	}

	.message-content {
		white-space: pre-wrap;
	}

	.message-time {
		font-size: 0.75rem;
		opacity: 0.8;
		text-align: right;
		margin-top: 4px;
	}

	.loading-wrapper {
		opacity: 0.8;
	}

	.loading-wrapper .message-content {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.el-icon-loading {
		font-size: 1.2em;
		animation: rotating 2s linear infinite;
	}

	@keyframes rotating {
		from {
			transform: rotate(0deg);
		}

		to {
			transform: rotate(360deg);
		}
	}
</style>