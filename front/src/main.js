import Vue from 'vue'
import App from './App.vue'
import axios from 'axios'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'

// 配置axios
Vue.prototype.$http = axios

// 可选：设置默认baseURL
// axios.defaults.baseURL = 'http://192.168.36.1:5000' // 您的API地址
axios.defaults.baseURL = 'http://127.0.0.1:5000' // 您的API地址

Vue.use(ElementUI)

new Vue({
	render: h => h(App),
}).$mount('#app')