# Quantum Mathematics - Lộ trình 65 ngày

Lộ trình toàn diện 65 ngày học về Toán học Lượng tử, Lập trình Quantum với Python & Qiskit, và Ứng dụng Quantum trong Credit Risk Management.

## 🎯 Tổng quan

Dự án này cung cấp một lộ trình học tập có cấu trúc từ cơ bản đến nâng cao về:

- **Fundamental (Ngày 1-15)**: Toán học cơ bản và lượng tử
- **Advanced (Ngày 16-35)**: Lập trình Quantum với Python & Qiskit
- **Credit Risk (Ngày 1-30)**: Ứng dụng Quantum trong Credit Risk Management

## 🚀 Tính năng

- **Giao diện hiện đại**: Thiết kế Just the Docs inspired
- **Navigation thông minh**: Sidebar với search và phân cấp rõ ràng
- **Responsive design**: Hoạt động tốt trên mọi thiết bị
- **Nội dung có cấu trúc**: 65 ngày học tập được tổ chức logic

## 🛠️ Cài đặt và chạy local

### Yêu cầu
- Ruby 2.7+ 
- Jekyll 4.0+
- Bundler

### Cài đặt

1. **Clone repository**:
```bash
git clone https://github.com/nguyenngocbinh/quantum-mathematics.git
cd quantum-mathematics
```

2. **Cài đặt dependencies**:
```bash
bundle install
```

3. **Chạy local server**:
```bash
bundle exec jekyll serve
```

4. **Mở trình duyệt**: http://localhost:4000

### Test CSS (không cần Jekyll)

Nếu bạn chỉ muốn xem CSS mới mà không cần cài đặt Jekyll:

1. Mở file `test.html` trong trình duyệt
2. File này chứa tất cả các thành phần UI để test

## 📁 Cấu trúc dự án

```
quantum-mathematics/
├── _layouts/          # Layout templates
├── _includes/         # Include files
├── assets/
│   └── css/          # CSS files
├── fundamental/       # Ngày 1-15
├── advanced/         # Ngày 16-35
├── credit-risk/      # Ngày 1-30
├── _config.yml       # Jekyll config
├── Gemfile          # Ruby dependencies
└── test.html        # Test file cho CSS
```

## 🎨 CSS Features

### Components
- **Sidebar Navigation**: Fixed sidebar với search
- **Cards**: Modern card design với hover effects
- **Callouts**: Info, warning, success, danger boxes
- **Badges**: Color-coded badges
- **Buttons**: Primary, secondary, outline variants
- **Tables**: Styled tables với hover effects
- **Progress bars**: Animated progress indicators

### Responsive Design
- **Desktop**: Full sidebar navigation
- **Tablet**: Collapsible sidebar
- **Mobile**: Hamburger menu với overlay

### Color Scheme
- **Primary**: Blue (#2563eb)
- **Secondary**: Gray (#6b7280)
- **Accent**: Red (#dc2626)
- **Success**: Green (#22c55e)
- **Warning**: Orange (#f59e0b)

## 🔄 GitHub Actions

Dự án sử dụng GitHub Actions để tự động build và deploy:

- **Trigger**: Push to main branch
- **Build**: Jekyll build với custom CSS
- **Deploy**: GitHub Pages

## 📝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📧 Liên hệ

- **Email**: nguyenngocbinhneu@gmail.com
- **GitHub**: [@nguyenngocbinh](https://github.com/nguyenngocbinh)

## 📄 License

MIT License - xem file LICENSE để biết thêm chi tiết.

---

**🎯 Tiến độ học tập**: Fundamental (1-15) → Advanced (16-35) → Credit Risk (1-30) 